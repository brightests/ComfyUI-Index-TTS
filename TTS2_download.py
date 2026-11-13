#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndexTTS-2 Model Download Script
自动下载所有 IndexTTS-2 所需的模型文件（基于 huggingface_hub）
支持断点续传、镜像加速（HF_ENDPOINT）、本地缓存（HF_HOME），并按项目要求放置到固定目录结构
"""

import os
import sys
from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


def _resolve_models_root() -> Path:
    """Return the ComfyUI models root (the parent of IndexTTS-2).

    Resolution order:
      1. INDEXTTS_MODELS_DIR environment variable (explicit override).
      2. folder_paths.models_dir when ComfyUI is already initialised, so
         --models-directory / --base-directory are honoured.
      3. <ComfyUI>/models derived from this script's location.

    Download scripts normally run standalone, where folder_paths is not loaded.
    Set INDEXTTS_MODELS_DIR in that case to match a custom --models-directory.
    Keep in sync with indextts2/model_loader.py and indextts2_5/model_loader.py.
    """
    override = os.environ.get("INDEXTTS_MODELS_DIR")
    if override:
        return Path(override).expanduser().resolve()

    # Never import ComfyUI here: comfy.cli_args parses sys.argv at import time.
    folder_paths = sys.modules.get("folder_paths")
    if folder_paths is not None:
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            return Path(models_dir).resolve()

    # .../ComfyUI/custom_nodes/ComfyUI-Index-TTS/TTS2_download.py -> .../ComfyUI/models
    return Path(__file__).resolve().parent.parent.parent / "models"


class ModelDownloader:
    def __init__(self):
        # 跟随 ComfyUI 的 --models-directory / --base-directory；独立运行时可用 INDEXTTS_MODELS_DIR 覆盖
        self.models_dir = _resolve_models_root() / "IndexTTS-2"
        # 创建目录
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 默认 endpoint（可在 ask_mirror_preference 中修改）
        self.endpoint_official = "https://huggingface.co"
        self.endpoint_mirror = "https://hf-mirror.com"  # 国内镜像
        self.current_endpoint = self.endpoint_official

        # 在模型目录下设置一个 Hugging Face 缓存目录，离线优先
        self.hf_home = self.models_dir / "hf_cache"
        os.environ.setdefault("HF_HOME", str(self.hf_home))
    
    def ask_mirror_preference(self):
        """询问是否使用国内镜像，并设置 HF_ENDPOINT 与缓存目录"""
        print("检测到您可能在中国大陆地区访问，是否使用国内镜像加速下载？")
        print("1. 使用官方地址 (huggingface.co)")
        print("2. 使用国内镜像 (hf-mirror.com) - 推荐")

        while True:
            choice = input("请选择 (1/2，默认为2): ").strip()
            if choice == "1":
                self.current_endpoint = self.endpoint_official
                print("已选择官方地址")
                break
            elif choice == "2" or choice == "":
                self.current_endpoint = self.endpoint_mirror
                print("已选择国内镜像")
                break
            else:
                print("请输入1或2")

        # 设置 HF_ENDPOINT 与 HF_HOME（在 Windows 下同样适用）
        os.environ["HF_ENDPOINT"] = self.current_endpoint
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        # 可选：启用更快的传输（若安装了 hf_transfer）
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    
    # 统一的下载方法集合（基于 huggingface_hub）
    def _snapshot(self, repo_id: str, allow_patterns: List[str], local_dir: Path):
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision="main",
            allow_patterns=allow_patterns,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    def _download_file(self, repo_id: str, filename: str, local_path: Path):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision="main",
                local_dir=str(local_path.parent),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            # hf_hub_download 会把文件放在 local_dir/filename
            # 确保最终路径为 local_path
            final_path = local_path
            if Path(cached_path) != final_path:
                # 当 local_dir 已经是目标目录时，cached_path 就等于 final_path
                pass
            return True
        except EntryNotFoundError:
            print(f"✗ 远端未找到文件: {repo_id}:{filename}")
            return False
    
    def download_all(self):
        """按固定目录结构下载所有所需模型文件"""
        print(f"\n{'='*50}")
        print("开始下载所有模型...")
        print(f"{'='*50}")

        success = True

        # 1) 基础模型文件（IndexTeam/IndexTTS-2 根目录下）
        # 按你的 TTS2模型路径.txt 中的列举进行下载
        print("\n[1/6] 下载基础模型 (IndexTeam/IndexTTS-2 根目录)...")
        base_files = [
            "bpe.model",
            "campplus_cn_common.bin",
            "config.yaml",
            "feat1.pt",
            "feat2.pt",
            "gpt.pth",
            "s2mel.pth",
            "wav2vec2bert_stats.pt",
        ]
        try:
            self._snapshot(
                repo_id="IndexTeam/IndexTTS-2",
                allow_patterns=base_files,
                local_dir=self.models_dir,
            )
            print("✓ 基础模型文件下载完成")
        except Exception as e:
            print(f"✗ 基础模型下载失败: {e}")
            success = False

        # 2) qwen0.6bemo4-merge 子目录
        print("\n[2/6] 下载 qwen0.6bemo4-merge 子目录...")
        try:
            self._snapshot(
                repo_id="IndexTeam/IndexTTS-2",
                allow_patterns=["qwen0.6bemo4-merge/*"],
                local_dir=self.models_dir,
            )
            print("✓ qwen0.6bemo4-merge 下载完成")
        except Exception as e:
            print(f"✗ qwen0.6bemo4-merge 下载失败: {e}")
            success = False

        # 3) semantic codec (amphion/MaskGCT) -> semantic_codec/model.safetensors
        print("\n[3/6] 下载 semantic codec (MaskGCT 语义编码器)...")
        try:
            target = self.models_dir / "semantic_codec" / "model.safetensors"
            # 正确放置：若 local_dir 指向 semantic_codec，则 filename 只需为文件名
            ok = self._download_file(
                repo_id="amphion/MaskGCT",
                filename="semantic_codec/model.safetensors",
                local_path=target,
            )
            # 兼容之前下载到 semantic_codec/semantic_codec/model.safetensors 的旧路径，自动修正
            wrong_nested = self.models_dir / "semantic_codec" / "semantic_codec" / "model.safetensors"
            if not target.exists() and wrong_nested.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    wrong_nested.rename(target)
                    print("已将 semantic codec 文件从嵌套目录移动到正确位置")
                except Exception:
                    pass
            # 若目标文件已在正确位置，标记为成功
            if target.exists():
                ok = True
            if ok:
                print("✓ semantic codec 下载完成")
            else:
                success = False
        except Exception as e:
            print(f"✗ semantic codec 下载失败: {e}")
            success = False

        # 4) CampPlus 说话人嵌入（若根目录已有 campplus_cn_common.bin 则跳过）
        print("\n[4/6] 确认 CampPlus 说话人嵌入...")
        try:
            campplus_local = self.models_dir / "campplus_cn_common.bin"
            if not campplus_local.exists():
                print("未发现本地 campplus_cn_common.bin，尝试从 funasr/campplus 下载...")
                ok = self._download_file(
                    repo_id="funasr/campplus",
                    filename="campplus_cn_common.bin",
                    local_path=campplus_local,
                )
                if ok:
                    print("✓ CampPlus 下载完成")
                else:
                    success = False
            else:
                print("已存在 campplus_cn_common.bin，跳过下载")
        except Exception as e:
            print(f"✗ CampPlus 下载失败: {e}")
            success = False

        # 5) w2v-bert-2.0 整仓（facebook/w2v-bert-2.0）
        print("\n[5/6] 下载 Wav2Vec2Bert 特征提取器 (facebook/w2v-bert-2.0)...")
        try:
            self._snapshot(
                repo_id="facebook/w2v-bert-2.0",
                allow_patterns=["*"],
                local_dir=self.models_dir / "w2v-bert-2.0",
            )
            print("✓ w2v-bert-2.0 下载完成")
        except Exception as e:
            print(f"✗ w2v-bert-2.0 下载失败: {e}")
            success = False

        # 6) BigVGAN 声码器（nvidia/bigvgan_v2_22khz_80band_256x）
        print("\n[6/6] 下载 BigVGAN 声码器 (nvidia/bigvgan_v2_22khz_80band_256x)...")
        try:
            self._snapshot(
                repo_id="nvidia/bigvgan_v2_22khz_80band_256x",
                allow_patterns=["*"],
                local_dir=self.models_dir / "bigvgan" / "bigvgan_v2_22khz_80band_256x",
            )
            print("✓ BigVGAN 下载完成")
        except Exception as e:
            print(f"✗ BigVGAN 下载失败: {e}")
            success = False

        return success
    
    def verify_downloads(self):
        """验证下载的文件"""
        print(f"\n{'='*50}")
        print("验证下载的文件...")
        print(f"{'='*50}")
        
        required_files = [
            "config.yaml",  # 基础模型文件
            "qwen0.6bemo4-merge",
            "semantic_codec/model.safetensors",
            "campplus_cn_common.bin",
            "w2v-bert-2.0",
            "bigvgan"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.models_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                print(f"✓ {file_path}")
        
        if missing_files:
            print(f"\n缺少以下文件:")
            for file_path in missing_files:
                print(f"✗ {file_path}")
            return False
        else:
            print(f"\n✓ 所有必需文件都已下载完成!")
            return True
    
    def run(self):
        """运行下载脚本"""
        print("IndexTTS-2 模型下载脚本")
        print("=" * 50)
        print(f"模型将下载到: {self.models_dir.absolute()}")
        
        # 询问镜像偏好
        self.ask_mirror_preference()
        
        try:
            ok = self.download_all()
        except KeyboardInterrupt:
            print("\n用户中断下载")
            sys.exit(1)
        except Exception as e:
            print(f"下载过程中出错: {e}")
            ok = False

        # 验证文件
        print(f"\n{'='*50}")
        print("下载完成报告")
        print(f"{'='*50}")
        if self.verify_downloads() and ok:
            print(f"\n🎉 所有模型下载完成! 模型路径: {self.models_dir.absolute()}")
        else:
            print(f"\n⚠️  部分文件可能缺失，请重新运行脚本或检查网络/镜像设置")

if __name__ == "__main__":
    try:
        downloader = ModelDownloader()
        downloader.run()
    except KeyboardInterrupt:
        print("\n下载已取消")
        sys.exit(1)
    except Exception as e:
        print(f"脚本运行出错: {e}")
        sys.exit(1)
