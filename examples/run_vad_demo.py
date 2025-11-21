from __future__ import annotations

import sys
from pathlib import Path

import torch

# 确保可以从 examples/ 下直接运行脚本时找到项目包
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemoasr2pytorch.vad.api import run_vad_on_waveform


def main() -> None:
    wav_path = REPO_ROOT / "test_long.wav"
    model_path = REPO_ROOT / "exports/frame_vad_multilingual_marblenet_v2.0.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    probs, segments = run_vad_on_waveform(model, str(wav_path))

    print(f"Frame probs shape: {probs.shape}")
    for i, seg in enumerate(segments):
        print(f"Segment {i}: {seg.start:.3f}s - {seg.end:.3f}s, mean_prob={seg.mean_prob:.3f}")


if __name__ == "__main__":
    main()
