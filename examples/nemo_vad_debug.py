from __future__ import annotations

"""
使用 NeMo 官方 EncDecFrameClassificationModel 的最简 VAD 调试脚本。

作用：
- 在独立的 NeMo 环境中，直接加载 nvidia/frame_vad_multilingual_marblenet_v2.0；
- 计算帧级 speech 概率，并按简单阈值规则生成语音段，方便和本项目的 VAD 结果做对比。
"""

import argparse
from pathlib import Path

import librosa
import torch
import nemo.collections.asr as nemo_asr


def frame_probs_to_segments(
    probs: torch.Tensor,
    frame_stride: float,
    threshold: float = 0.5,
    min_speech_duration: float = 0.2,
    min_silence_duration: float = 0.1,
):
    """
    将帧级概率转换为语音段列表，逻辑与本项目 VAD 近似：
    - 连续 probs >= threshold 的帧视为语音；
    - 丢弃过短语音段；
    - 合并中间静音过短的相邻段。
    """
    probs_np = probs.cpu().numpy()
    T = probs_np.shape[0]
    speech_mask = probs_np >= threshold

    segments = []
    in_seg = False
    seg_start = 0

    def idx2time(i: int) -> float:
        return i * frame_stride

    # 1) 连续语音帧切段
    for i in range(T):
        if speech_mask[i] and not in_seg:
            in_seg = True
            seg_start = i
        elif not speech_mask[i] and in_seg:
            in_seg = False
            seg_end = i
            start_t = idx2time(seg_start)
            end_t = idx2time(seg_end)
            if end_t - start_t >= min_speech_duration:
                mean_prob = float(probs_np[seg_start:seg_end].mean())
                segments.append((start_t, end_t, mean_prob))

    if in_seg:
        seg_end = T
        start_t = idx2time(seg_start)
        end_t = idx2time(seg_end)
        if end_t - start_t >= min_speech_duration:
            mean_prob = float(probs_np[seg_start:seg_end].mean())
            segments.append((start_t, end_t, mean_prob))

    # 2) 合并中间静音很短的相邻段
    if not segments:
        return []

    merged = [segments[0]]
    for start, end, mp in segments[1:]:
        prev_start, prev_end, prev_mp = merged[-1]
        gap = start - prev_end
        if gap < min_silence_duration:
            total_dur = (prev_end - prev_start) + (end - start)
            if total_dur > 0:
                mean_prob = (
                    prev_mp * (prev_end - prev_start) + mp * (end - start)
                ) / total_dur
            else:
                mean_prob = max(prev_mp, mp)
            merged[-1] = (prev_start, end, mean_prob)
        else:
            merged.append((start, end, mp))

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal NeMo MarbleNet VAD debug script.\n"
            "Loads nvidia/frame_vad_multilingual_marblenet_v2.0 via NeMo and prints speech segments."
        )
    )
    parser.add_argument(
        "--wav",
        type=str,
        required=True,
        help="Path to input WAV file (will be resampled to 16kHz mono).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Speech probability threshold in [0,1] (default: 0.8).",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)

    # 1) 加载 NeMo VAD 模型
    # 关键：strict=False，忽略较新 NeMo 类上多出来但 checkpoint 中不存在的 loss.weight 等键。
    vad_model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
        model_name="nvidia/frame_vad_multilingual_marblenet_v2.0",
        strict=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vad_model = vad_model.to(device)
    vad_model.eval()

    # 从配置中读取帧移（秒），后续把帧索引换算为时间
    frame_stride = float(vad_model._cfg.preprocessor.window_stride)

    # 2) 加载音频，重采样到 16kHz 单声道
    sig, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    sig_t = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)  # [1, T]
    sig_len = torch.tensor([sig_t.shape[1]], dtype=torch.long)

    # 3) 前向推理，得到帧级 logits
    with torch.no_grad():
        logits = vad_model(
            input_signal=sig_t.to(device),
            input_signal_length=sig_len.to(device),
        )

    # 4) 取 speech 类概率（logits -> softmax）
    # EncDecFrameClassificationModel 通常输出 [B, T, C]
    if logits.dim() == 3:
        frame_logits = logits[0]  # [T, C]
    elif logits.dim() == 2:
        frame_logits = logits
    else:
        raise RuntimeError(f"Unexpected logits shape: {logits.shape}")

    probs = torch.softmax(frame_logits, dim=-1)[:, 1]  # class 1 = speech，形状 [T]

    # 5) 阈值化 + 合并得到语音段
    segments = frame_probs_to_segments(
        probs,
        frame_stride=frame_stride,
        threshold=args.threshold,
        min_speech_duration=0.2,
        min_silence_duration=0.1,
    )

    print(f"Frame probs shape: {probs.shape}")
    for i, (start, end, mp) in enumerate(segments):
        print(f"Segment {i}: {start:.3f}s - {end:.3f}s, mean_prob={mp:.3f}")


if __name__ == "__main__":
    main()
