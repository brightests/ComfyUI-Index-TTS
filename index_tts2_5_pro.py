# coding=utf-8
"""
IndexTTS-2.5 Pro Node - 多角色小说朗读节点。

在 IndexTTS-2.5 上复刻 2.0 的 Pro 多角色编排，并补齐 2.5 独有能力：
- lang（多语言）/ duration_factor（语速）/ text_normalization / interval_silence
- 文本内联发音标注 <行|XING2> 保持可用（不会被当成角色标签切分）
- 结构化文本同时支持 <Narrator>/<Character1..5> 与 <林黛玉> 等任意角色名
- 按角色预提取 conditioning 并复用，避免小说每一段都重新编码音色

输出与 2.0 Pro 保持一致：audio / seed / Subtitle / SimplifiedSubtitle。
"""

import json
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .index_tts2_5_nodes import (
    LANG_CHOICES,
    _GLOBAL_ENGINE,
    _GLOBAL_LOADER,
    _IndexTTS25BaseMixin,
)
from .indextts2_5.utils import save_temp_wav

try:
    import torchaudio

    _HAS_TORCHAUDIO = True
except Exception:  # pragma: no cover - 缺少 torchaudio 时自动退回路径缓存模式
    torchaudio = None
    _HAS_TORCHAUDIO = False

# 角色标签：排除含 "|" 的 2.5 发音标注（<行|XING2> / <word|W ER1 D>）
_ROLE_TAG_RE = re.compile(r"<([^<>|\n]{1,40})>")
_BUILTIN_ROLE_RE = re.compile(r"^character\s*([1-9]\d*)$", re.IGNORECASE)
_NARRATOR_ALIASES = {"narrator", "narration", "旁白", "旁白音", "叙述"}
_SENTENCE_SPLIT_RE = re.compile(r"([,，.。!！?？;；])")


class IndexTTS25ProNode(_IndexTTS25BaseMixin):
    """IndexTTS-2.5 多角色小说朗读：结构化文本 + 每个角色一套参考音色。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "structured_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "<Narrator>这是一段旁白。<Character1>你好，我是角色一。<林黛玉>今日风有些大。",
                    },
                ),
                "narrator_audio": ("AUDIO",),
                "lang": (LANG_CHOICES, {"default": "ZH"}),
                "duration_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "character1_audio": ("AUDIO",),
                "character2_audio": ("AUDIO",),
                "character3_audio": ("AUDIO",),
                "character4_audio": ("AUDIO",),
                "character5_audio": ("AUDIO",),
                # 中文/自定义角色名 -> Narrator / Character1..5（或 1..5）
                "role_map": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": '{"林黛玉": "Character1", "宝玉": 2}',
                    },
                ),
                # 开启后，未映射的角色名按出现顺序自动占用空闲的角色音频端口
                "auto_assign_roles": ("BOOLEAN", {"default": False}),
                # 全局情感参考音频（不提供时，每个角色用自己的音色作为情感参考）
                "emotion_audio": ("AUDIO",),
                "emotion_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                # 2.5 生成参数
                "do_sample_mode": (["off", "on"], {"default": "on"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "length_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "max_mel_tokens": ("INT", {"default": 1500, "min": 50, "max": 1815, "step": 10}),
                "max_tokens_per_sentence": ("INT", {"default": 120, "min": 0, "max": 600, "step": 5}),
                "interval_silence_ms": ("INT", {"default": 200, "min": 0, "max": 2000, "step": 50}),
                "text_normalization": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "cache_control": ("DICT", {"default": None}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "STRING", "STRING")
    RETURN_NAMES = ("audio", "seed", "Subtitle", "SimplifiedSubtitle")
    FUNCTION = "generate_multi_voice_speech"
    CATEGORY = "audio"
    DESCRIPTION = (
        "IndexTTS-2.5 多角色小说朗读：<Narrator>/<CharacterN>/<任意角色名>，"
        "支持 2.5 多语言、语速与发音标注，按角色复用音色特征"
    )

    def __init__(self):
        # 与 2.5 其他节点共用全局 loader，避免重复加载模型权重
        self.loader = _GLOBAL_LOADER
        self.engine = _GLOBAL_ENGINE

    # ------------------------------------------------------------------ 文本与角色

    @staticmethod
    def _parse_structured_text(structured_text: str) -> List[Tuple[str, str]]:
        """切分角色标签；含 | 的 2.5 发音标注保留在正文里，不当作角色分隔。"""
        text = structured_text or ""
        matches = list(_ROLE_TAG_RE.finditer(text))
        if not matches:
            return [("Narrator", text.strip())] if text.strip() else []

        segments: List[Tuple[str, str]] = []
        prefix = text[: matches[0].start()].strip()
        if prefix:
            segments.append(("Narrator", prefix))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[match.end() : end].strip()
            if body:
                segments.append((match.group(1).strip(), body))
        return segments

    @staticmethod
    def _load_role_map(raw: Any) -> Dict[str, Any]:
        if raw is None or not str(raw).strip():
            return {}
        try:
            data = json.loads(raw)
        except Exception as exc:
            print(f"[IndexTTS2.5 Pro] role_map 不是合法 JSON，已忽略: {exc}")
            return {}
        if not isinstance(data, dict):
            print("[IndexTTS2.5 Pro] role_map 必须是 JSON 对象，已忽略")
            return {}
        return {str(k).strip().strip("<>").strip(): v for k, v in data.items()}

    @staticmethod
    def _normalize_role_target(value: Any) -> Optional[str]:
        """把 role_map 的值归一到 Narrator / CharacterN。"""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return f"Character{value}" if 1 <= value <= 5 else None
        text = str(value).strip().strip("<>").strip()
        builtin = _BUILTIN_ROLE_RE.match(text)
        if builtin:
            return f"Character{int(builtin.group(1))}"
        if text.lower() in _NARRATOR_ALIASES:
            return "Narrator"
        if text.isdigit() and 1 <= int(text) <= 5:
            return f"Character{int(text)}"
        return None

    @classmethod
    def _resolve_roles(cls, segments, role_map, ports, auto_assign=False):
        """返回 [(canonical_role, display_name, text), ...]，未匹配角色回退旁白音色。"""
        available = [f"Character{i}" for i in range(1, 6) if ports.get(f"Character{i}") is not None]
        assigned = set()
        resolved = []
        for name, body in segments:
            key = (name or "").strip()
            canonical = None
            if key.lower() in _NARRATOR_ALIASES:
                canonical = "Narrator"
            else:
                builtin = _BUILTIN_ROLE_RE.match(key)
                if builtin:
                    canonical = f"Character{int(builtin.group(1))}"
                if canonical is None and key in role_map:
                    canonical = cls._normalize_role_target(role_map[key])
                if canonical is None and auto_assign:
                    for candidate in available:
                        if candidate not in assigned:
                            canonical = candidate
                            assigned.add(candidate)
                            print(f"[IndexTTS2.5 Pro] 角色 <{key}> 自动分配到 {candidate}")
                            break
            if canonical is None:
                print(f"[IndexTTS2.5 Pro] 角色 <{key}> 未配置音频，使用旁白音色")
                canonical = "Narrator"
            elif ports.get(canonical) is None:
                print(f"[IndexTTS2.5 Pro] {canonical} 未连接参考音频，回退旁白音色")
                canonical = "Narrator"
            resolved.append((canonical, key or "Narrator", body))
        return resolved

    # ------------------------------------------------------------------ 音色复用

    @staticmethod
    @torch.no_grad()
    def _extract_voice(tts, wave: np.ndarray, sr: int) -> Dict[str, torch.Tensor]:
        """复用 2.5 Extract Voice 节点的特征提取，产出 CPU conditioning。"""
        if not _HAS_TORCHAUDIO:
            raise RuntimeError("torchaudio 不可用")
        audio = torch.as_tensor(wave, dtype=torch.float32)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim > 2:
            audio = audio.reshape(1, -1)
        elif audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio[:, : int(sr * 15)]

        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        inputs = tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(tts.device)
        attention_mask = inputs["attention_mask"].to(tts.device)
        spk_cond = tts.get_emb(input_features, attention_mask)

        ref_mel = tts.mel_fn(audio_22k.to(tts.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(ref_mel.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = tts.campplus_model(feat.unsqueeze(0))

        s2mel_prompt = tts.s2mel.models["length_regulator"](
            spk_cond,
            ylens=ref_target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]

        return {
            "spk_cond": spk_cond.detach().cpu(),
            "style": style.detach().cpu(),
            "s2mel_prompt": s2mel_prompt.detach().cpu(),
            "ref_mel": ref_mel.detach().cpu(),
            "emo_cond": spk_cond.detach().cpu().clone(),
        }

    @staticmethod
    def _prime_role_cache(tts, voice: Dict[str, torch.Tensor], key: str, external_emo: bool):
        """把该角色的 conditioning 注入上游 infer 缓存，避免重新编码。"""
        device = tts.device
        tts.cache_spk_cond = voice["spk_cond"].to(device)
        tts.cache_s2mel_style = voice["style"].to(device)
        tts.cache_s2mel_prompt = voice["s2mel_prompt"].to(device)
        tts.cache_mel = voice["ref_mel"].to(device)
        tts.cache_spk_audio_prompt = key
        if not external_emo:
            # 无外部情感音频时，上游会用 spk_audio_prompt 作为 emo_audio_prompt
            tts.cache_emo_cond = voice["emo_cond"].to(device)
            tts.cache_emo_audio_prompt = key

    # ------------------------------------------------------------------ 字幕

    @staticmethod
    def _seconds_to_time_format(seconds: float) -> str:
        minutes = int(seconds // 60)
        remaining = seconds % 60
        whole = int(remaining)
        millis = int((remaining - whole) * 1000)
        return f"{minutes}:{whole:02d}.{millis:03d}"

    @classmethod
    def _build_subtitles(cls, subtitle_data):
        subtitle_json = json.dumps(
            [
                {
                    "id": item["id"],
                    "字幕": item["字幕"],
                    "start": cls._seconds_to_time_format(item["start"]),
                    "end": cls._seconds_to_time_format(item["end"]),
                }
                for item in subtitle_data
            ],
            ensure_ascii=False,
            indent=2,
        )

        lines = []
        for item in subtitle_data:
            text = item["字幕"]
            parts = _SENTENCE_SPLIT_RE.split(text)
            sentences = (
                [s + punct for s, punct in zip(parts[::2], parts[1::2] + [""])]
                if len(parts) > 1
                else [text]
            )
            sentences = [s for s in sentences if s.strip()] or [text]
            total = max(0.0, item["end"] - item["start"])
            per_sentence = total / len(sentences)
            for i, sentence in enumerate(sentences):
                start = item["start"] + i * per_sentence
                lines.append(f">> {cls._seconds_to_time_format(start)}-{cls._seconds_to_time_format(start + per_sentence)}")
                lines.append(f">> {sentence}")
        return subtitle_json, "\n".join(lines)

    # ------------------------------------------------------------------ 主流程

    def generate_multi_voice_speech(
        self,
        structured_text: str,
        narrator_audio,
        lang: str = "ZH",
        duration_factor: float = 1.0,
        character1_audio=None,
        character2_audio=None,
        character3_audio=None,
        character4_audio=None,
        character5_audio=None,
        role_map="",
        auto_assign_roles: bool = False,
        emotion_audio=None,
        emotion_weight: float = 0.8,
        do_sample_mode: str = "on",
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        length_penalty: float = 0.0,
        max_mel_tokens: int = 1500,
        max_tokens_per_sentence: int = 120,
        interval_silence_ms: int = 200,
        text_normalization: bool = True,
        seed: int = 0,
        cache_control=None,
    ):
        segments = self._parse_structured_text(structured_text)
        if not segments:
            raise ValueError("结构化文本为空，没有可合成的段落")

        ports = {"Narrator": narrator_audio}
        for i, audio in enumerate(
            [character1_audio, character2_audio, character3_audio, character4_audio, character5_audio], 1
        ):
            if audio is not None:
                ports[f"Character{i}"] = audio

        mapping = self._load_role_map(role_map)
        resolved = self._resolve_roles(segments, mapping, ports, auto_assign_roles)
        print(f"[IndexTTS2.5 Pro] 解析到 {len(resolved)} 个段落，{len({r[0] for r in resolved})} 个角色")

        tts = self.loader.get_tts(use_qwen_emo=False)
        external_emo = emotion_audio is not None
        emo_path = save_temp_wav(self._process_audio_input(emotion_audio)) if external_emo else None

        # 每个实际用到的角色只写一次 WAV，并尽量预提取 conditioning
        sources = {}
        for role in {item[0] for item in resolved}:
            wave, sr = self._process_audio_input(ports[role])
            path = save_temp_wav((wave, sr))
            key = f"indextts25pro-voice://{role}"
            voice = None
            try:
                voice = self._extract_voice(tts, wave, sr)
            except Exception as exc:
                print(f"[IndexTTS2.5 Pro] {role} 音色预提取失败，改用固定 WAV 路径: {exc}")
            sources[role] = {"path": path, "key": key, "voice": voice}
        if any(src["voice"] is not None for src in sources.values()):
            print(f"[IndexTTS2.5 Pro] 已为 {sum(s['voice'] is not None for s in sources.values())} 个角色预提取音色特征")

        audio_segments: List[Tuple[np.ndarray, int]] = []
        subtitle_data = []
        current_time = 0.0

        for index, (role, display, text) in enumerate(resolved):
            # 每段用可复现但互不相同的种子
            segment_seed = (int(seed) + index) % (2**32)
            torch.manual_seed(segment_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(segment_seed)

            src = sources[role]
            if src["voice"] is not None:
                self._prime_role_cache(tts, src["voice"], src["key"], external_emo)
                prompt = src["key"]
            else:
                prompt = src["path"]

            print(f"[IndexTTS2.5 Pro] [{index + 1}/{len(resolved)}] <{display}> -> {role}: {text[:40]}...")
            try:
                sr, wave, _ = self.engine.generate(
                    text=text,
                    reference_audio=None,
                    reference_audio_path=prompt,
                    lang=lang,
                    duration_factor=duration_factor,
                    do_sample=(do_sample_mode == "on"),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    max_mel_tokens=max_mel_tokens,
                    max_tokens_per_sentence=max_tokens_per_sentence,
                    interval_silence=interval_silence_ms,
                    text_normalization=text_normalization,
                    emo_ref_audio=None,
                    emo_audio_path=emo_path,
                    emo_weight=float(emotion_weight),
                    seed=segment_seed,
                    return_subtitles=False,
                )
            except Exception as exc:
                print(f"[IndexTTS2.5 Pro] 段落 <{display}> 合成失败，已跳过: {exc}")
                traceback.print_exc()
                continue

            duration = len(wave) / float(sr)
            subtitle_data.append(
                {
                    "id": display,
                    "字幕": text,
                    "start": current_time,
                    "end": current_time + duration,
                }
            )
            current_time += duration
            audio_segments.append((wave, sr))

        if not audio_segments:
            raise RuntimeError("没有成功生成任何音频段落")

        sample_rate = audio_segments[0][1]
        waves = []
        for wave, sr in audio_segments:
            if sr != sample_rate:
                if _HAS_TORCHAUDIO:
                    wave = torchaudio.functional.resample(
                        torch.from_numpy(wave), sr, sample_rate
                    ).numpy()
                else:
                    print(f"[IndexTTS2.5 Pro] 警告: 段落采样率 {sr} 与 {sample_rate} 不一致")
            waves.append(wave)
        concatenated = np.concatenate(waves, axis=0)

        wave_tensor = torch.tensor(concatenated, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        audio_output = {"waveform": wave_tensor, "sample_rate": int(sample_rate)}
        subtitle_json, simplified = self._build_subtitles(subtitle_data)

        self._maybe_unload(cache_control)
        print(f"[IndexTTS2.5 Pro] 完成，共 {len(audio_segments)} 段，总时长 {current_time:.2f} 秒")
        return (audio_output, seed, subtitle_json, simplified)
