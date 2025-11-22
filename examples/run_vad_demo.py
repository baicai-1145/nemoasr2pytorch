from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保可以从 examples/ 下直接运行脚本时找到项目包
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemoasr2pytorch.vad.api import load_default_frame_vad_model, run_vad_on_waveform


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MarbleNet VAD demo and print detected speech segments."
    )
    parser.add_argument(
        "--wav",
        type=str,
        default=str(REPO_ROOT / "test_long.wav"),
        help="Path to input WAV file (default: test_long.wav).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="VAD speech probability threshold in [0,1] (default: 0.5).",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav)

    model = load_default_frame_vad_model()
    probs, segments = run_vad_on_waveform(model, str(wav_path), threshold=args.threshold)

    print(f"Frame probs shape: {probs.shape}")
    for i, seg in enumerate(segments):
        print(f"Segment {i}: {seg.start:.3f}s - {seg.end:.3f}s, mean_prob={seg.mean_prob:.3f}")


if __name__ == "__main__":
    main()
