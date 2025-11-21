from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

import torch

# 允许从 examples/ 目录直接运行
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemoasr2pytorch.conversion.parakeet_tdt import load_parakeet_tdt_from_nemo
from nemoasr2pytorch.conversion.vad import load_frame_vad_from_nemo


def _extract_nemo(nemo_path: Path, extract_dir: Path) -> None:
    """将 .nemo 文件解压到指定目录（支持 zip / tar）。"""
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 先尝试按 tar 处理（当前用户的 .nemo 属于这种情况）
    # 若不是 tar，再退回 zip 处理。
    try:
        if tarfile.is_tarfile(nemo_path):
            with tarfile.open(nemo_path, "r:*") as tf:
                tf.extractall(extract_dir)
            return
    except tarfile.TarError:
        pass

    if zipfile.is_zipfile(nemo_path):
        with zipfile.ZipFile(nemo_path, "r") as zf:
            zf.extractall(extract_dir)
        return

    raise RuntimeError(f"Unsupported .nemo archive format: {nemo_path}")


def _find_required_files(root: Path) -> tuple[Path, Path]:
    """在解压目录中查找 model_config.yaml 和 model_weights.ckpt。"""
    config_candidates = list(root.rglob("model_config.yaml"))
    ckpt_candidates = list(root.rglob("model_weights.ckpt"))

    if not config_candidates:
        raise FileNotFoundError(f"model_config.yaml not found under {root}")
    if not ckpt_candidates:
        raise FileNotFoundError(f"model_weights.ckpt not found under {root}")

    # 一般情况下各只有一个，若有多个则取最浅层级的一个
    config_path = sorted(config_candidates, key=lambda p: len(p.parts))[0]
    ckpt_path = sorted(ckpt_candidates, key=lambda p: len(p.parts))[0]
    return config_path, ckpt_path


def export_from_nemo(kind: str, nemo_path: Path, output_path: Path, extract_dir: Path | None = None) -> None:
    """
    从 NeMo .nemo 或已解压目录构建等价的 PyTorch 推理模型并保存为 .pt。

    - 若 nemo_path 指向 .nemo 文件，则会解压到 extract_dir（未指定时默认使用同名子目录）。
    - 若 nemo_path 指向目录，则视为已解压好的根目录，不再重复解压。
    """
    nemo_path = nemo_path.resolve()

    # 情况 1：传入的是已经解压好的目录
    if nemo_path.is_dir():
        root = nemo_path
    else:
        if not nemo_path.is_file():
            raise FileNotFoundError(f".nemo file not found: {nemo_path}")

        if extract_dir is None:
            extract_dir = nemo_path.parent / nemo_path.stem
        extract_dir = extract_dir.resolve()

        # 总是执行一次解压，确保 model_config.yaml / model_weights.ckpt 都存在
        _extract_nemo(nemo_path, extract_dir)
        root = extract_dir

    config_path, ckpt_path = _find_required_files(root)

    # 构建等价 PyTorch 模型（固定 CPU FP32，保存时不带设备依赖）
    if kind == "vad":
        model = load_frame_vad_from_nemo(
            config_path=config_path,
            ckpt_path=ckpt_path,
            device="cpu",
            dtype=torch.float32,
        )
    elif kind == "parakeet_tdt":
        # parakeet 需要 tokenizer 目录，直接用解压目录
        model = load_parakeet_tdt_from_nemo(
            config_path=config_path,
            ckpt_path=ckpt_path,
            tokenizer_dir=extract_dir,
            device="cpu",
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 保存整个模型对象，便于后续直接 torch.load 使用
    torch.save(model, output_path)
    print(f"Saved PyTorch model to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 NeMo .nemo 文件解压并构建等价的 PyTorch 推理模型（保存为 .pt）。"
    )
    parser.add_argument(
        "--kind",
        required=True,
        choices=["vad", "parakeet_tdt"],
        help="模型类型：'vad' 对应 MarbleNet VAD，'parakeet_tdt' 对应 parakeet-tdt-0.6b-v2。",
    )
    parser.add_argument("--nemo", required=True, type=str, help="NeMo .nemo 模型文件路径。")
    parser.add_argument("--output", required=True, type=str, help="导出的 .pt 文件路径。")
    parser.add_argument(
        "--extract-dir",
        type=str,
        default=None,
        help="可选，.nemo 解压目录（默认使用与 .nemo 同名的子目录）。",
    )
    args = parser.parse_args()

    nemo_path = Path(args.nemo)
    output_path = Path(args.output)
    extract_dir = Path(args.extract_dir) if args.extract_dir is not None else None

    export_from_nemo(kind=args.kind, nemo_path=nemo_path, output_path=output_path, extract_dir=extract_dir)


if __name__ == "__main__":
    main()
