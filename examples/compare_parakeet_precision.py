from __future__ import annotations

from pathlib import Path
import sys
import time

import torch

# 允许从 examples/ 目录直接运行
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemoasr2pytorch.asr.api import (
    load_default_parakeet_tdt_model,
    load_parakeet_tdt_fp16,
    transcribe,
    transcribe_amp,
)


def _print_cuda_mem(prefix: str) -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"{prefix} | allocated={allocated:.1f}MB, reserved={reserved:.1f}MB")


def main() -> None:
    wav_path = REPO_ROOT / "test_long.wav"
    if not wav_path.is_file():
        wav_path = REPO_ROOT / "test_01.wav"

    if not torch.cuda.is_available():
        print("CUDA 不可用，只演示 FP32 CPU 推理。")
        model = load_default_parakeet_tdt_model(device="cpu")
        t0 = time.time()
        text = transcribe(model, str(wav_path))
        t1 = time.time()
        print(f"[CPU FP32] time={t1 - t0:.2f}s, text={text}")
        return

    # FP32
    torch.cuda.empty_cache()
    model_fp32 = load_default_parakeet_tdt_model(device="cuda", dtype=torch.float32)
    _print_cuda_mem("[FP32] after load")
    t0 = time.time()
    text_fp32 = transcribe(model_fp32, str(wav_path))
    t1 = time.time()
    _print_cuda_mem("[FP32] after infer")

    # FP16
    torch.cuda.empty_cache()
    model_fp16 = load_parakeet_tdt_fp16(device="cuda")
    _print_cuda_mem("[FP16] after load")
    t2 = time.time()
    text_fp16 = transcribe_amp(model_fp16, str(wav_path))
    t3 = time.time()
    _print_cuda_mem("[FP16] after infer")

    print(f"[FP32] time={t1 - t0:.2f}s, text={text_fp32}")
    print(f"[FP16] time={t3 - t2:.2f}s, text={text_fp16}")


if __name__ == "__main__":
    main()

