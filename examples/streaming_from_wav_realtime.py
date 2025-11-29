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
        "--cpu-only",
        action="store_true",
        help="Force VAD and ASR to run on CPU. In this mode precision is effectively fp32.",
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

    # 选择设备与精度（支持强制 CPU）
    device: str | torch.device
    precision = args.precision
    if args.cpu_only:
        device = "cpu"
        if precision != "fp32":
            print(
                "[nemoasr2pytorch] --cpu-only ignores low-precision setting "
                f"(precision={precision}), forcing fp32 on CPU."
            )
            precision = "fp32"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
    committed_text = ""
    pending_trim_sec = 0.0
    prev_words: list[str] = []
    while start < total_len:
        end = min(start + chunk_samples, total_len)
        chunk = waveform[start:end]
        is_last = end >= total_len
        is_first = chunk_idx == 0

        # 如有待应用的前缀裁剪（按上一轮确认的句子边界），先裁剪内部 waveform 上下文
        if pending_trim_sec > 0.0:
            pipeline.trim_prefix_seconds(pending_trim_sec)
            pending_trim_sec = 0.0
            prev_words = []

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

        # 基于最近两次的 word 序列，按句号等标点以及“下一个词已出现”的规则，
        # 决定是否可以将前缀句子“定稿”，并在下一轮裁剪对应的音频上下文。
        try:
            word_offsets = pipeline.last_word_offsets
        except AttributeError:
            word_offsets = []

        curr_words = [w["word"] for w in word_offsets]

        if has_speech and prev_words and curr_words:
            # 计算按 word 的最长公共前缀长度
            lcp_len = 0
            for w_prev, w_curr in zip(prev_words, curr_words):
                if w_prev != w_curr:
                    break
                lcp_len += 1

            # 在 LCP 内寻找最后一个以句号/问号/感叹号结尾且后面还有词的词，作为可定稿句子的结尾
            punctuations = (".", "!", "?", "。", "！", "？")
            candidate_idx = None
            for idx in range(lcp_len):
                if curr_words[idx].endswith(punctuations) and idx + 1 < len(curr_words):
                    candidate_idx = idx

            if candidate_idx is not None:
                cut_word = word_offsets[candidate_idx]
                cut_end_sec = float(cut_word["end"])

                # 记录待裁剪的前缀时长（在下一轮 stream_step 前应用），
                # 文本前缀则立刻作为 committed_text 固定下来。
                pending_trim_sec = cut_end_sec
                committed_text = " ".join(w["word"] for w in word_offsets[: candidate_idx + 1])
                # 一旦确定句子边界，重置 prev_words，使后续 LCP 在新的上下文上重新计算
                prev_words = []
            else:
                prev_words = curr_words
        else:
            if has_speech and curr_words:
                prev_words = curr_words

        t1 = time.perf_counter()

        if args.realtime:
            # 尝试让字幕与音频播放时间对齐：等待到该 chunk 实际播放结束附近再打印
            elapsed = t1 - t0
            chunk_dur = (end - start) / target_sr
            remain = chunk_dur - elapsed
            if remain > 0:
                time.sleep(remain)

        # 展示文本：已定稿前缀 + 当前未定稿部分
        if committed_text:
            remaining_words = curr_words
            remaining_text = " ".join(remaining_words) if remaining_words else ""
            if remaining_text:
                display_text = committed_text + " " + remaining_text
            else:
                display_text = committed_text
        else:
            display_text = partial_text

        print(
            f"[chunk {chunk_idx}] {start/target_sr:.2f}s - {end/target_sr:.2f}s "
            f"({(end-start)/target_sr:.2f}s) "
            f"{'[speech]' if has_speech else '[silence]'} partial: {display_text}"
        )

        start = end
        chunk_idx += 1

    final_text = pipeline.model.tokenizer.decode(pipeline._all_token_ids)
    print("\n=== Final Transcription (streaming, simulated mic) ===")
    print(final_text)


if __name__ == "__main__":
    main()
