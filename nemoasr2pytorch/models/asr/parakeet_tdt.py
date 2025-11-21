from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from nemoasr2pytorch.audio.melspectrogram import AudioToMelSpectrogramPreprocessor
from nemoasr2pytorch.decoding.tdt_greedy import GreedyTDTDecoder, TDTGreedyConfig
from nemoasr2pytorch.models.asr.conformer import ConformerEncoder
from nemoasr2pytorch.models.asr.rnnt import RNNTDecoder, RNNTJoint
from nemoasr2pytorch.text.tokenizer import TextTokenizer


@dataclass
class ParakeetTDTConfig:
    sample_rate: int
    blank_id: int
    durations: List[int]


class ParakeetTDTModel(nn.Module):
    """Parakeet-tdt-0.6b-v2 推理模型（纯 PyTorch）。"""

    def __init__(
        self,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        encoder: ConformerEncoder,
        decoder: RNNTDecoder,
        joint: RNNTJoint,
        tokenizer: TextTokenizer,
        cfg: ParakeetTDTConfig,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.tokenizer = tokenizer
        self.cfg = cfg

        # 默认的 TDT 贪心解码器（单样本）
        self._tdt_decoder = GreedyTDTDecoder(
            decoder=decoder,
            joint=joint,
            cfg=TDTGreedyConfig(
                blank_id=cfg.blank_id,
                durations=cfg.durations,
                max_symbols_per_step=10,
            ),
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def blank_id(self) -> int:
        return self.cfg.blank_id

    @property
    def sample_rate(self) -> int:
        return self.cfg.sample_rate

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveform: [T] or [B,T]，采样率为 cfg.sample_rate。
        Returns:
            enc_out: [B, D, T_enc]
            enc_len: [B]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        B, T = waveform.shape
        lengths = torch.full((B,), T, dtype=torch.long, device=waveform.device)
        feats, feat_len = self.preprocessor(waveform, lengths)
        enc_out, enc_len = self.encoder(feats, feat_len)
        return enc_out, enc_len

    @torch.no_grad()
    def greedy_decode(
        self,
        enc_out: torch.Tensor,
        enc_len: torch.Tensor,
    ) -> List[int]:
        """调用 TDT label-looping 贪心解码器（单样本版本）。"""
        return self._tdt_decoder.decode(enc_out, enc_len)

    @torch.no_grad()
    def transcribe_ids(self, waveform: torch.Tensor) -> List[int]:
        enc_out, enc_len = self.encode(waveform)
        return self.greedy_decode(enc_out, enc_len)

    @torch.no_grad()
    def transcribe(self, waveform: torch.Tensor) -> str:
        ids = self.transcribe_ids(waveform)
        return self.tokenizer.decode(ids)
