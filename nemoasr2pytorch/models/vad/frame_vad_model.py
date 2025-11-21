from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from nemoasr2pytorch.audio.melspectrogram import AudioToMelSpectrogramPreprocessor
from nemoasr2pytorch.models.vad.marblenet import MarbleNetEncoder


class FrameVADDecoder(nn.Module):
    """简单的线性分类器，将 MarbleNet 输出映射到二分类 logits。"""

    def __init__(self, in_features: int, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        logits = self.linear(x)
        return logits  # [B, T, num_classes]


class FrameVADModel(nn.Module):
    """组合 Mel 预处理 + MarbleNet 编码器 + 线性解码器的帧级 VAD 模型。"""

    def __init__(
        self,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        encoder: MarbleNetEncoder,
        decoder: FrameVADDecoder,
        labels: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.labels = labels or ["0", "1"]

    @torch.no_grad()
    def forward(
        self, audio_signal: torch.Tensor, length: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_signal: [B, T] waveform
            length: [B] 样本长度（单位：sample）；若为 None，则根据 T 推出。
        Returns:
            frame_logits: [B, T', num_classes]
            frame_probs:  [B, T'] 语音概率（取 index=1）
        """
        if audio_signal.dim() == 1:
            audio_signal = audio_signal.unsqueeze(0)

        if length is None:
            length = torch.full(
                size=(audio_signal.size(0),),
                fill_value=audio_signal.size(1),
                dtype=torch.long,
                device=audio_signal.device,
            )

        features, feat_len = self.preprocessor(audio_signal, length)
        encoded, enc_len = self.encoder(features)
        logits = self.decoder(encoded)  # [B, T', C]

        probs = torch.softmax(logits, dim=-1)[..., 1]
        return logits, probs

    @torch.no_grad()
    def predict_frame_probs(
        self, audio_signal: torch.Tensor, length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """仅返回帧级语音概率 [B, T']。"""
        _, probs = self.forward(audio_signal, length)
        return probs

    @property
    def frame_stride(self) -> float:
        """返回帧移（秒）。"""
        return self.preprocessor.frame_stride

