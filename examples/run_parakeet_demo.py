from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许从 examples/ 目录直接运行
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemoasr2pytorch.asr.api import (
    load_default_parakeet_tdt_model,
    load_parakeet_tdt_fp16,
    load_parakeet_tdt_bf16,
    transcribe,
    transcribe_amp,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Parakeet-TDT demo with language/precision presets."
    )
    parser.add_argument(
        "--wav",
        type=str,
        default=str(REPO_ROOT / "test_01.wav"),
        help="Path to input WAV file (default: test_01.wav).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["EN", "EU"],
        default="EN",
        help="Language preset: EN -> v2 (English), EU -> v3 (multilingual).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Precision preset: fp32 / fp16 / bf16.",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav)

    if args.precision == "fp32":
        model = load_default_parakeet_tdt_model(lang=args.lang)
        text = transcribe(model, str(wav_path))
    elif args.precision == "fp16":
        model = load_parakeet_tdt_fp16(lang=args.lang)
        text = transcribe_amp(model, str(wav_path))
    else:  # bf16
        model = load_parakeet_tdt_bf16(lang=args.lang)
        text = transcribe_amp(model, str(wav_path))

    print(f"lang={args.lang} precision={args.precision} transcription: {text}")


if __name__ == "__main__":
    main()
