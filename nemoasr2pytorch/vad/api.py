from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch

from nemoasr2pytorch.models.vad.frame_vad_model import FrameVADModel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAD_PT = REPO_ROOT / "exports/frame_vad_multilingual_marblenet_v2.0.pt"


def load_default_frame_vad_model(
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> FrameVADModel:
    """
    加载默认的 Frame_VAD_Multilingual_MarbleNet_v2.0 模型（仅支持 .pt）。

    要先使用 examples/export_from_nemo_to_pt.py 将 .nemo 导出为 exports/frame_vad_marblenet.pt。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if DEFAULT_VAD_PT.is_file():
        model = torch.load(DEFAULT_VAD_PT, map_location=device, weights_only=False)
        model.to(device=device, dtype=dtype)
        model.eval()
        return model

    raise FileNotFoundError(
        f"Default VAD .pt model not found: {DEFAULT_VAD_PT}. "
        "Please export it from the original .nemo using examples/export_from_nemo_to_pt.py."
    )


@dataclass
class VadSegment:
    start: float  # seconds
    end: float    # seconds
    mean_prob: float


def _to_waveform_tensor(x: Union[str, np.ndarray, torch.Tensor], target_sr: int) -> Tuple[torch.Tensor, int]:
    if isinstance(x, torch.Tensor):
        if x.dim() == 2 and x.size(0) == 1:
            x = x.squeeze(0)
        if x.dim() != 1:
            raise ValueError("Only mono 1D waveform tensor is supported.")
        return x.float(), target_sr

    if isinstance(x, np.ndarray):
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 1:
            raise ValueError("Only mono 1D numpy waveform is supported.")
        return torch.from_numpy(x.astype("float32")), target_sr

    # string path
    import torchaudio

    wav_path = Path(x)
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    sig, sr = torchaudio.load(str(wav_path))
    if sig.size(0) > 1:
        sig = sig.mean(dim=0, keepdim=True)
    sig = sig.squeeze(0)
    if sr != target_sr:
        sig = torchaudio.functional.resample(sig, orig_freq=sr, new_freq=target_sr)
        sr = target_sr
    return sig.float(), sr


def _frame_probs_to_segments(
    probs: torch.Tensor,
    frame_stride: float,
    threshold: float = 0.5,
    min_speech_duration: float = 0.2,
    min_silence_duration: float = 0.1,
) -> List[VadSegment]:
    """
    将帧级概率转换为语音段列表（简单阈值 + 合并）。
    probs: [T]
    """
    probs_np = probs.cpu().numpy()
    T = probs_np.shape[0]
    speech_mask = probs_np >= threshold

    segments: List[VadSegment] = []
    in_segment = False
    seg_start = 0

    def index_to_time(idx: int) -> float:
        return idx * frame_stride

    for i in range(T):
        if speech_mask[i] and not in_segment:
            in_segment = True
            seg_start = i
        elif not speech_mask[i] and in_segment:
            in_segment = False
            seg_end = i
            start_t = index_to_time(seg_start)
            end_t = index_to_time(seg_end)
            if end_t - start_t >= min_speech_duration:
                mean_prob = float(probs_np[seg_start:seg_end].mean())
                segments.append(VadSegment(start=start_t, end=end_t, mean_prob=mean_prob))

    if in_segment:
        seg_end = T
        start_t = index_to_time(seg_start)
        end_t = index_to_time(seg_end)
        if end_t - start_t >= min_speech_duration:
            mean_prob = float(probs_np[seg_start:seg_end].mean())
            segments.append(VadSegment(start=start_t, end=end_t, mean_prob=mean_prob))

    # 合并过短静音间隔的相邻段
    if not segments:
        return segments

    merged: List[VadSegment] = [segments[0]]
    for seg in segments[1:]:
        gap = seg.start - merged[-1].end
        if gap < min_silence_duration:
            # merge
            total_dur = (merged[-1].end - merged[-1].start) + (seg.end - seg.start)
            if total_dur > 0:
                mean_prob = (
                    merged[-1].mean_prob * (merged[-1].end - merged[-1].start)
                    + seg.mean_prob * (seg.end - seg.start)
                ) / total_dur
            else:
                mean_prob = max(merged[-1].mean_prob, seg.mean_prob)
            merged[-1] = VadSegment(start=merged[-1].start, end=seg.end, mean_prob=mean_prob)
        else:
            merged.append(seg)

    return merged


def run_vad_on_waveform(
    model: FrameVADModel,
    audio: Union[str, np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    min_speech_duration: float = 0.2,
    min_silence_duration: float = 0.1,
) -> Tuple[torch.Tensor, List[VadSegment]]:
    """
    对单条音频运行 VAD，返回帧级概率和语音段列表。
    """
    device = next(model.parameters()).device
    waveform, sr = _to_waveform_tensor(audio, target_sr=model.preprocessor.sample_rate)
    waveform = waveform.to(device=device)

    probs = model.predict_frame_probs(waveform.unsqueeze(0))[0]  # [T']
    segments = _frame_probs_to_segments(
        probs,
        frame_stride=model.frame_stride,
        threshold=threshold,
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
    )
    return probs, segments
