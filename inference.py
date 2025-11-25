from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio

# 允许从仓库根目录直接运行脚本时找到项目包
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemoasr2pytorch.asr.api import (  # type: ignore[import]
    load_default_parakeet_tdt_model,
    load_parakeet_tdt_fp16,
    load_parakeet_tdt_bf16,
    transcribe,
    transcribe_amp,
)
from nemoasr2pytorch.vad.api import (  # type: ignore[import]
    load_default_frame_vad_model,
    run_vad_on_waveform,
)


# 模型级别的最长单段时长限制（秒）
MODEL_MAX_SEC = {
    "EN": 24 * 60.0,        # v2: 24 分钟
    "EU": 3 * 60.0 * 60.0,  # v3: 3 小时
}


def load_waveform_mono(path: Path, target_sr: int) -> torch.Tensor:
    """
    从文件加载单通道 waveform，并重采样到 target_sr。
    返回形状 [T] 的 float32 张量。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    sig, sr = torchaudio.load(str(path))
    if sig.size(0) > 1:
        sig = sig.mean(dim=0, keepdim=True)
    sig = sig.squeeze(0)
    if sr != target_sr:
        sig = torchaudio.functional.resample(sig, orig_freq=sr, new_freq=target_sr)
    return sig.float()


def group_vad_segments(
    segments,
    min_seg: float,
    max_seg: float,
) -> List[Tuple[float, float]]:
    """
    将 VAD 段按时间合并成较长的区间，使得每段时长尽量在 [min_seg, max_seg] 之间。
    输入 segments 是 VadSegment 列表，至少包含 .start / .end（单位：秒）。
    返回 [(start, end), ...]，单位：秒。
    """
    if not segments:
        return []

    # 先按 max_seg 进行粗分组
    groups: List[Tuple[float, float]] = []
    cur_start = float(segments[0].start)
    cur_end = float(segments[0].end)

    for seg in segments[1:]:
        s = float(seg.start)
        e = float(seg.end)
        if e - cur_start <= max_seg:
            # 继续扩展当前段
            cur_end = e
        else:
            groups.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    groups.append((cur_start, cur_end))

    # 再合并过短的段，使其尽量 ≥ min_seg（最后一段可能仍短于 min_seg）
    merged: List[Tuple[float, float]] = []
    for g in groups:
        if not merged:
            merged.append(g)
            continue
        prev_start, prev_end = merged[-1]
        prev_len = prev_end - prev_start
        if prev_len < min_seg:
            # 将当前段并入上一个段
            merged[-1] = (prev_start, g[1])
        else:
            merged.append(g)

    return merged


def split_without_vad(duration_sec: float, max_seg: float) -> List[Tuple[float, float]]:
    """
    在不使用 VAD 的情况下，按固定长度 max_seg 将整段音频切分。
    返回 [(start, end), ...]，单位：秒。
    """
    segments: List[Tuple[float, float]] = []
    start = 0.0
    while start < duration_sec:
        end = min(start + max_seg, duration_sec)
        segments.append((start, end))
        start = end
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-form ASR inference script: optional VAD + Parakeet-TDT."
    )
    parser.add_argument(
        "--wav",
        type=str,
        required=True,
        help="Path to input WAV file.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["EN", "EU"],
        default="EU",
        help="Language preset: EN -> parakeet-tdt-0.6b-v2 (English), "
        "EU -> parakeet-tdt-0.6b-v3 (multilingual).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Precision preset for Parakeet ASR: fp32 / fp16 / bf16.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force all models to run on CPU (ASR + VAD). "
        "In this mode precision is effectively fp32.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD; run ASR on fixed-length chunks only.",
    )
    parser.add_argument(
        "--min-seg",
        type=float,
        default=10.0,
        help="Minimum target segment duration (seconds) when using VAD (default: 10).",
    )
    parser.add_argument(
        "--max-seg",
        type=float,
        default=60.0,
        help="Maximum target segment duration (seconds) per ASR chunk (default: 60).",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.8,
        help="VAD speech probability threshold in [0,1]. "
        "Higher -> fewer, more confident speech segments (default: 0.5).",
    )
    parser.add_argument(
        "--debug-time",
        action="store_true",
        help="Print simple time profiling of VAD vs ASR.",
    )
    parser.add_argument(
        "--with-word-ts",
        action="store_true",
        help="Print word-level timestamps for each ASR chunk "
        "(requires Parakeet model with `transcribe_with_word_timestamps`).",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav)
    lang = args.lang.upper()

    # CPU-only 模式下强制使用 fp32，避免无意义的低精度 CPU 计算
    precision = args.precision
    if args.cpu_only and precision != "fp32":
        print(
            f"[nemoasr2pytorch] CPU-only mode ignores precision={precision}, "
            "forcing fp32."
        )
        precision = "fp32"

    device_override: str | None = "cpu" if args.cpu_only else None

    t_total_start = time.perf_counter()
    t_model_start = time.perf_counter()

    # 选择 ASR 模型 + 推理函数（根据精度）
    if precision == "fp32":
        asr_model = load_default_parakeet_tdt_model(lang=lang, device=device_override)
        asr_infer = transcribe
    elif precision == "fp16":
        asr_model = load_parakeet_tdt_fp16(device=device_override, lang=lang)
        asr_infer = transcribe_amp
    else:  # bf16
        asr_model = load_parakeet_tdt_bf16(device=device_override, lang=lang)
        asr_infer = transcribe_amp
    model_load_time = time.perf_counter() - t_model_start

    target_sr = asr_model.sample_rate
    max_model_sec = MODEL_MAX_SEC[lang]

    # 生效的 max_seg 不能超过模型极限
    effective_max_seg = min(args.max_seg, max_model_sec)
    if args.max_seg > max_model_sec:
        print(
            f"[nemoasr2pytorch] Requested max_seg={args.max_seg:.1f}s exceeds model limit "
            f"{max_model_sec/60:.1f} min for lang={lang}, clamping to {effective_max_seg:.1f}s."
        )

    # 加载 waveform（按 ASR 采样率）
    t_audio_start = time.perf_counter()
    waveform = load_waveform_mono(wav_path, target_sr=target_sr)
    audio_load_time = time.perf_counter() - t_audio_start
    total_duration = waveform.numel() / float(target_sr)

    vad_time = 0.0
    asr_time = 0.0
    chunk_overhead_time = 0.0

    # 构建切分区间（单位：秒）
    if args.no_vad:
        # 不使用 VAD，按固定长度切分
        segments_sec = split_without_vad(total_duration, effective_max_seg)
        print(
            f"[nemoasr2pytorch] VAD disabled, splitting audio of {total_duration:.1f}s "
            f"into {len(segments_sec)} chunks (<= {effective_max_seg:.1f}s each)."
        )
    else:
        t_vad_start = time.perf_counter()

        # 使用 VAD，按语音段聚合
        vad_model = load_default_frame_vad_model(device=device_override)
        vad_device = next(vad_model.parameters()).device

        # 简化处理：假定 VAD 与 ASR 采样率一致，否则提示不支持
        vad_sr = vad_model.preprocessor.sample_rate
        if vad_sr != target_sr:
            raise RuntimeError(
                f"VAD sample_rate={vad_sr} != ASR sample_rate={target_sr}, "
                "current inference script assumes they are equal."
            )

        # 直接将 waveform 作为张量传入 VAD（run_vad_on_waveform 自身不会重采样张量）
        _, vad_segments = run_vad_on_waveform(
            vad_model,
            waveform,
            threshold=args.vad_threshold,
        )

        # VAD 用完后主动释放 GPU 显存
        del vad_model
        if vad_device.type == "cuda":  # pragma: no cover - 取决于是否有 GPU
            torch.cuda.empty_cache()
        if not vad_segments:
            print("[nemoasr2pytorch] No speech detected by VAD, nothing to transcribe.")
            return

        segments_sec = group_vad_segments(
            vad_segments,
            min_seg=args.min_seg,
            max_seg=effective_max_seg,
        )
        vad_time = time.perf_counter() - t_vad_start

        print(
            f"[nemoasr2pytorch] VAD produced {len(vad_segments)} raw segments, "
            f"grouped into {len(segments_sec)} chunks (target {args.min_seg:.1f}-{effective_max_seg:.1f}s)."
        )

    # 对每个区间切片 waveform 并运行 ASR
    texts: List[str] = []
    for idx, (start_s, end_s) in enumerate(segments_sec):
        start_idx = int(round(start_s * target_sr))
        end_idx = int(round(end_s * target_sr))
        chunk = waveform[start_idx:end_idx]

        if chunk.numel() == 0:
            continue

        # 为 ASR 模型提供 1D waveform 张量
        t_chunk_start = time.perf_counter()
        t_asr_start = t_chunk_start

        if args.with_word_ts and hasattr(asr_model, "transcribe_with_word_timestamps"):
            # 直接使用模型的带时间戳接口（内部会重复 encode 一遍，简单但清晰）
            asr_device = next(asr_model.parameters()).device
            text, word_offsets = asr_model.transcribe_with_word_timestamps(  # type: ignore[attr-defined]
                chunk.to(device=asr_device)
            )
        else:
            text = asr_infer(asr_model, chunk)
            word_offsets = None

        t_asr_end = time.perf_counter()
        asr_time += t_asr_end - t_asr_start

        # chunk 内除 ASR 之外的开销（打印、字符串拼接等）
        chunk_overhead_time += time.perf_counter() - t_chunk_start - (t_asr_end - t_asr_start)

        texts.append(text.strip())
        print(
            f"[chunk {idx}] {start_s:.2f}s - {end_s:.2f}s "
            f"({end_s - start_s:.2f}s) [{lang}/{precision}]: {text}"
        )

        if args.with_word_ts and word_offsets:
            for w in word_offsets:
                print(
                    f"  -> {w['word']!r}: {w['start']:.2f}s - {w['end']:.2f}s "
                    f"(frames {w['start_offset']}-{w['end_offset']})"
                )

    full_text = " ".join(t for t in texts if t)
    print("\n=== Final Transcription ===")
    print(full_text)

    if args.debug_time:
        total_time = time.perf_counter() - t_total_start
        other = total_time - vad_time - asr_time - model_load_time - audio_load_time - chunk_overhead_time
        print(
            "\n[profiling]\n"
            f"  model_load_time   = {model_load_time:.2f}s\n"
            f"  audio_load_time   = {audio_load_time:.2f}s\n"
            f"  vad_time          = {vad_time:.2f}s\n"
            f"  asr_time          = {asr_time:.2f}s\n"
            f"  chunk_overhead    = {chunk_overhead_time:.2f}s\n"
            f"  other             = {other:.2f}s\n"
            f"  total             = {total_time:.2f}s"
        )


if __name__ == "__main__":
    main()
