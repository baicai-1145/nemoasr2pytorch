from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemoasr2pytorch.asr.api import (  # type: ignore[import]
    load_default_parakeet_tdt_model,
    load_parakeet_tdt_fp16,
    load_parakeet_tdt_bf16,
)
from nemoasr2pytorch.asr.streaming.pipeline import BufferedParakeetPipeline, StreamingConfig  # type: ignore[import]
from nemoasr2pytorch.vad.api import (  # type: ignore[import]
    load_default_frame_vad_model,
    run_vad_on_waveform,
)


def load_waveform(path: Path, target_sr: int) -> torch.Tensor:
    sig, sr = torchaudio.load(str(path))
    if sig.size(0) > 1:
        sig = sig.mean(dim=0, keepdim=True)
    sig = sig.squeeze(0)
    if sr != target_sr:
        sig = torchaudio.functional.resample(sig, orig_freq=sr, new_freq=target_sr)
    return sig.float()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate microphone streaming by feeding test_long.wav in chunks to the "
            "Parakeet streaming pipeline (optionally in real time)."
        )
    )
    parser.add_argument("--wav", type=str, default=str(REPO_ROOT / "test_long.wav"), help="Path to WAV file.")
    parser.add_argument(
        "--lang",
        type=str,
        choices=["EN", "EU"],
        default="EN",
        help="Language preset: EN -> parakeet-tdt-0.6b-v2, EU -> parakeet-tdt-0.6b-v3.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Precision preset for Parakeet ASR: fp32 / fp16 / bf16.",
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=8.0,
        help="Chunk size in seconds to simulate microphone streaming (default: 8).",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep between chunks to simulate real-time audio (chunk duration minus ASR time).",
    )
    parser.add_argument(
        "--play-audio",
        action="store_true",
        help="Attempt to play the WAV audio while streaming (requires sounddevice).",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.8,
        help="VAD speech probability threshold in [0,1]. "
        "Chunks with no detected speech will be skipped for ASR (default: 0.8).",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav)
    lang = args.lang.upper()

    # 选择设备与精度
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = args.precision

    if precision == "fp32":
        asr_model = load_default_parakeet_tdt_model(lang=lang, device=device)
    elif precision == "fp16":
        asr_model = load_parakeet_tdt_fp16(device=device, lang=lang)
    else:
        asr_model = load_parakeet_tdt_bf16(device=device, lang=lang)

    target_sr = asr_model.sample_rate
    waveform = load_waveform(wav_path, target_sr=target_sr)

    # 加载 VAD 模型（用于按 chunk 判断是否有语音）
    vad_model = load_default_frame_vad_model(device=device)
    vad_sr = vad_model.preprocessor.sample_rate
    if vad_sr != target_sr:
        raise RuntimeError(
            f"VAD sample_rate={vad_sr} != ASR sample_rate={target_sr}, "
            "streaming_from_wav_realtime.py assumes they are equal."
        )

    # 构建 streaming pipeline
    stream_cfg = StreamingConfig(
        sample_rate=target_sr,
        chunk_size=args.chunk_sec,
        left_padding_size=0.0,
        right_padding_size=0.0,
        stop_history_eou_ms=-1,
    )
    pipeline = BufferedParakeetPipeline(model=asr_model, cfg=stream_cfg, device=device)
    pipeline.reset_state()

    # 可选：播放音频
    if args.play_audio:
        try:
            import sounddevice as sd  # type: ignore

            # sounddevice 期望 numpy 数组
            sd.play(waveform.cpu().numpy(), samplerate=target_sr)
        except Exception as e:  # pragma: no cover - 依赖本地环境
            print(f"[nemoasr2pytorch] Failed to play audio: {e}")

    total_len = waveform.numel()
    chunk_samples = int(args.chunk_sec * target_sr) if args.chunk_sec > 0 else total_len
    if chunk_samples <= 0:
        chunk_samples = total_len

    print(
        f"[nemoasr2pytorch] Streaming from {wav_path.name} "
        f"({total_len / target_sr:.2f}s) in chunks of {args.chunk_sec:.2f}s "
        f"on device={device} precision={precision}."
    )

    start = 0
    chunk_idx = 0
    last_partial_text = ""
    while start < total_len:
        end = min(start + chunk_samples, total_len)
        chunk = waveform[start:end]
        is_last = end >= total_len
        is_first = chunk_idx == 0

        t0 = time.perf_counter()
        # 先用 VAD 判断该 chunk 是否包含语音
        probs, segments = run_vad_on_waveform(
            vad_model,
            chunk,
            threshold=args.vad_threshold,
        )
        has_speech = len(segments) > 0

        if has_speech:
            partial_text = pipeline.stream_step(chunk, is_last=is_last, is_first=is_first)
            last_partial_text = partial_text
        else:
            partial_text = last_partial_text

        t1 = time.perf_counter()

        if args.realtime:
            # 尝试让字幕与音频播放时间对齐：等待到该 chunk 实际播放结束附近再打印
            elapsed = t1 - t0
            chunk_dur = (end - start) / target_sr
            remain = chunk_dur - elapsed
            if remain > 0:
                time.sleep(remain)

        print(
            f"[chunk {chunk_idx}] {start/target_sr:.2f}s - {end/target_sr:.2f}s "
            f"({(end-start)/target_sr:.2f}s) "
            f"{'[speech]' if has_speech else '[silence]'} partial: {partial_text}"
        )

        start = end
        chunk_idx += 1

    final_text = pipeline.model.tokenizer.decode(pipeline._all_token_ids)
    print("\n=== Final Transcription (streaming, simulated mic) ===")
    print(final_text)


if __name__ == "__main__":
    main()
