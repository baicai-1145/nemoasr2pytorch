from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torchaudio

from nemoasr2pytorch.asr.streaming.pipeline import BufferedParakeetPipeline, StreamingConfig
from nemoasr2pytorch.asr.api import load_default_parakeet_tdt_model


def build_parakeet_streaming_pipeline(
    lang: str = "EN",
    device: Optional[str] = None,
    chunk_size: float = 8.0,
    left_padding_size: float = 0.0,
    right_padding_size: float = 0.0,
    stop_history_eou_ms: int = -1,
) -> BufferedParakeetPipeline:
    """
    构建一个基于 Parakeet 的简化 streaming pipeline。
    """
    model = load_default_parakeet_tdt_model(device=device, lang=lang)
    cfg = StreamingConfig(
        sample_rate=model.sample_rate,
        chunk_size=chunk_size,
        left_padding_size=left_padding_size,
        right_padding_size=right_padding_size,
        stop_history_eou_ms=stop_history_eou_ms,
    )
    return BufferedParakeetPipeline(model=model, cfg=cfg, device=device or "cuda" if torch.cuda.is_available() else "cpu")


def streaming_transcribe(
    wav_path: str | Path,
    lang: str = "EN",
    device: Optional[str] = None,
    chunk_size: float = 8.0,
) -> str:
    """
    使用 streaming pipeline 对单个 WAV 文件进行转写。
    """
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    pipeline = build_parakeet_streaming_pipeline(lang=lang, device=device, chunk_size=chunk_size)

    sig, sr = torchaudio.load(str(wav_path))
    if sig.size(0) > 1:
        sig = sig.mean(dim=0, keepdim=True)
    sig = sig.squeeze(0)

    if sr != pipeline.sample_rate:
        sig = torchaudio.functional.resample(sig, orig_freq=sr, new_freq=pipeline.sample_rate)

    return pipeline.run_streaming(sig)

