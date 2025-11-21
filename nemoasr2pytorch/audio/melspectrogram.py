from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import librosa  # type: ignore
except Exception as e:  # pragma: no cover - import error surfaced at runtime
    librosa = None


CONSTANT = 1e-5


def _normalize_batch(
    x: torch.Tensor, seq_len: torch.Tensor, normalize_type: Optional[str]
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if normalize_type is None:
        return x, None, None

    x_mean = None
    x_std = None
    if normalize_type == "per_feature":
        batch_size = x.shape[0]
        max_time = x.shape[2]

        time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
        valid_mask = time_steps < seq_len.unsqueeze(1)
        x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(axis=2)
        x_mean_denominator = valid_mask.sum(axis=1)
        x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)

        # subtract 1 in the denominator to correct for the bias
        x_std = torch.sqrt(
            torch.sum(torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0) ** 2, axis=2)
            / (x_mean_denominator.unsqueeze(1) - 1.0)
        )
        x_std = x_std.masked_fill(x_std.isnan(), 0.0)
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, : seq_len[i].item()].mean()
            x_std[i] = x[i, :, : seq_len[i].item()].std()
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1), x_mean, x_std
    else:
        # unsupported normalize type – return as-is
        return x, None, None


def _splice_frames(x: torch.Tensor, frame_splicing: int) -> torch.Tensor:
    """Stacks frames together across feature dim.

    input:  [B, F, T]
    output: [B, F * frame_splicing, T]
    """
    if frame_splicing <= 1:
        return x

    seq = [x]
    for n in range(1, frame_splicing):
        seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
    return torch.cat(seq, dim=1)


class FilterbankFeatures(nn.Module):
    """近似 NeMo FilterbankFeatures 的 Mel 频谱特征抽取器（仅保留推理路径）。"""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_window_size: int = 320,
        n_window_stride: int = 160,
        window: str = "hann",
        normalize: Optional[str] = "per_feature",
        n_fft: Optional[int] = None,
        preemph: Optional[float] = 0.97,
        nfilt: int = 64,
        lowfreq: int = 0,
        highfreq: Optional[int] = None,
        log: bool = True,
        log_zero_guard_type: str = "add",
        log_zero_guard_value: float | str = 2**-24,
        dither: float = CONSTANT,
        pad_to: int | str = 2,
        frame_splicing: int = 1,
        exact_pad: bool = False,
        pad_value: float = 0.0,
        mag_power: float = 2.0,
        nb_augmentation_prob: float = 0.0,
        nb_max_freq: int = 4000,
        mel_norm: str = "slaney",
    ) -> None:
        super().__init__()

        if librosa is None:
            raise ImportError("librosa is required for FilterbankFeatures but is not installed.")

        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(n_window_size, int)
            or not isinstance(n_window_stride, int)
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise ValueError("n_window_size and n_window_stride must be positive integers.")

        self.sample_rate = sample_rate
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.exact_pad = exact_pad
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if exact_pad else None

        torch_windows = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
            "blackman": torch.blackman_window,
            "bartlett": torch.bartlett_window,
            "none": None,
        }
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
        self.register_buffer("window", window_tensor)

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2

        fb = librosa.filters.mel(
            sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq, norm=mel_norm
        )
        filterbanks = torch.tensor(fb, dtype=torch.float32).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        # 最大长度和 padding 仅在需要 pad_to>0 时使用
        self.max_length = None
        self.pad_value = pad_value
        self.mag_power = mag_power

        if log_zero_guard_type not in ("add", "clamp"):
            raise ValueError("log_zero_guard_type must be 'add' or 'clamp'.")
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value = log_zero_guard_value

        self.nb_augmentation_prob = nb_augmentation_prob
        self.nb_max_freq = nb_max_freq

    @property
    def filter_banks(self) -> torch.Tensor:
        return self.fb

    def _log_zero_guard_value(self, x: torch.Tensor) -> float:
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            if self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            raise ValueError("log_zero_guard_value must be float, 'tiny' or 'eps'")
        return float(self.log_zero_guard_value)

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False if self.exact_pad else True,
            window=self.window.to(dtype=torch.float32, device=x.device) if self.window is not None else None,
            return_complex=True,
            pad_mode="constant",
        )

    def get_seq_len(self, seq_len: torch.Tensor) -> torch.Tensor:
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length)
        return seq_len.to(dtype=torch.long)

    def forward(
        self, x: torch.Tensor, seq_len: torch.Tensor, linear_spec: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """输入: x [B, T], seq_len [B] (样本点数); 输出: 特征 [B, F, T'], 长度 [B]."""

        seq_len_time = seq_len
        seq_len = self.get_seq_len(seq_len)

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant").squeeze(
                1
            )

        # dither（只在训练模式启用）
        if self.training and self.dither > 0:
            x = x + self.dither * torch.randn_like(x)

        # pre-emphasis
        if self.preemph is not None:
            timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~timemask, 0.0)

        # STFT + 幅度谱
        with torch.amp.autocast(x.device.type, enabled=False):
            spec = self._stft(x)
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1))

        # 幂谱
        if self.mag_power != 1.0:
            spec = spec.pow(self.mag_power)

        if linear_spec:
            return spec, seq_len

        with torch.amp.autocast(spec.device.type, enabled=False):
            fb = self.fb.to(spec.dtype)
            spec = torch.matmul(fb, spec)

        if self.log:
            guard = self._log_zero_guard_value(spec)
            if self.log_zero_guard_type == "add":
                spec = torch.log(spec + guard)
            else:
                spec = torch.log(torch.clamp(spec, min=guard))

        # 帧拼接
        spec = _splice_frames(spec, self.frame_splicing)

        # 归一化
        spec, _, _ = _normalize_batch(spec, seq_len, self.normalize)

        # 时间维度 pad 到 pad_to 的倍数
        max_len = spec.size(-1)
        mask = torch.arange(max_len, device=spec.device)
        mask = mask.repeat(spec.size(0), 1) >= seq_len.unsqueeze(1)
        spec = spec.masked_fill(mask.unsqueeze(1), self.pad_value)

        if self.pad_to == "max" and self.max_length is not None:
            spec = nn.functional.pad(spec, (0, self.max_length - spec.size(-1)), value=self.pad_value)
        elif isinstance(self.pad_to, int) and self.pad_to > 0:
            pad_amt = spec.size(-1) % self.pad_to
            if pad_amt != 0:
                spec = nn.functional.pad(spec, (0, self.pad_to - pad_amt), value=self.pad_value)

        return spec, seq_len


class AudioToMelSpectrogramPreprocessor(nn.Module):
    """简化版 NeMo AudioToMelSpectrogramPreprocessor，仅保留推理需要的路径。"""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_size: float = 0.025,
        window_stride: float = 0.01,
        window: str = "hann",
        normalize: Optional[str] = None,
        n_fft: Optional[int] = 512,
        features: int = 80,
        frame_splicing: int = 1,
        dither: float = 1.0e-5,
        pad_to: int | str = 2,
        pad_value: float = 0.0,
        mag_power: float = 2.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate

        n_window_size = int(window_size * sample_rate)
        n_window_stride = int(window_stride * sample_rate)

        self.featurizer = FilterbankFeatures(
            sample_rate=sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            nfilt=features,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            pad_value=pad_value,
            mag_power=mag_power,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "AudioToMelSpectrogramPreprocessor":
        return cls(
            sample_rate=cfg.get("sample_rate", 16000),
            window_size=cfg.get("window_size", 0.025),
            window_stride=cfg.get("window_stride", 0.01),
            window=cfg.get("window", "hann"),
            normalize=cfg.get("normalize", None),
            n_fft=cfg.get("n_fft", 512),
            features=cfg.get("features", 80),
            frame_splicing=cfg.get("frame_splicing", 1),
            dither=cfg.get("dither", 1.0e-5),
            pad_to=cfg.get("pad_to", 2),
        )

    def forward(self, input_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_signal: [B, T] waveform in float32.
            length: [B] number of samples in each waveform.
        Returns:
            processed_signal: [B, F, T']
            processed_length: [B] number of frames.
        """

        features, feature_lengths = self.featurizer(input_signal, length)
        return features, feature_lengths

    @property
    def frame_stride(self) -> float:
        """以秒为单位的帧移（由 hop_length / sample_rate 决定）。"""
        return float(self.featurizer.hop_length) / float(self.sample_rate)

