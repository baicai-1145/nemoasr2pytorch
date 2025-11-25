from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import string

import torch
import torch.nn as nn

from nemoasr2pytorch.audio.melspectrogram import AudioToMelSpectrogramPreprocessor
from nemoasr2pytorch.decoding.tdt_greedy import GreedyTDTDecoder, TDTDecodeResult, TDTGreedyConfig
from nemoasr2pytorch.models.asr.conformer import ConformerEncoder
from nemoasr2pytorch.models.asr.rnnt import RNNTDecoder, RNNTJoint
from nemoasr2pytorch.text.tokenizer import TextTokenizer


@dataclass
class ParakeetTDTConfig:
    sample_rate: int
    blank_id: int
    durations: List[int]


class ParakeetTDTModel(nn.Module):
    """Parakeet-tdt-0.6b-v2/v3 推理模型（纯 PyTorch）。"""

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

        # 确保特征 dtype 与编码器权重 dtype 一致（特别是在 FP16/BF16 推理时）
        enc_dtype = next(self.encoder.parameters()).dtype
        feats = feats.to(dtype=enc_dtype)

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
        # 将 waveform 移到与模型相同的 device / dtype
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        waveform = waveform.to(device=device, dtype=dtype)

        enc_out, enc_len = self.encode(waveform)
        return self.greedy_decode(enc_out, enc_len)

    @torch.no_grad()
    def transcribe(self, waveform: torch.Tensor) -> str:
        ids = self.transcribe_ids(waveform)
        return self.tokenizer.decode(ids)

    # ---- 带时间戳的转写 ----

    @torch.no_grad()
    def transcribe_with_word_timestamps(self, waveform: torch.Tensor) -> Tuple[str, List[Dict[str, float]]]:
        """
        返回与 NeMo 类似的 word-level 时间戳信息。

        Returns:
            text: 转写文本
            words: List[dict]，每个元素大致形如：
                {
                    "word": "moment.",
                    "start_offset": 103,   # encoder 帧 index
                    "end_offset": 109,     # encoder 帧 index
                    "start": 8.24,         # 秒
                    "end": 8.72,           # 秒
                }
        """
        # 将 waveform 移到与模型相同的 device / dtype
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        waveform = waveform.to(device=device, dtype=dtype)

        enc_out, enc_len = self.encode(waveform)
        decode_res: TDTDecodeResult = self._tdt_decoder.decode_with_timestamps(enc_out, enc_len)

        token_ids = decode_res.token_ids
        token_starts = decode_res.token_starts
        token_durs = decode_res.token_durations

        # TDT 时间轴是在编码器输出帧上，实际时间步长 = mel 帧步长 * subsampling_factor
        subsampling_factor = getattr(self.encoder.cfg, "subsampling_factor", 1)
        frame_stride = self.preprocessor.frame_stride * float(subsampling_factor)
        word_offsets = self._bpe_tokens_to_word_offsets(token_ids, token_starts, token_durs, frame_stride)
        text = self.tokenizer.decode(token_ids)
        return text, word_offsets

    def _bpe_tokens_to_word_offsets(
        self,
        token_ids: List[int],
        token_starts: List[int],
        token_durs: List[int],
        frame_stride: float,
    ) -> List[Dict[str, float]]:
        """
        基于 SentencePiece BPE token 序列聚合出 word-level 时间戳。

        简化规则（与 SentencePiece 常用用法对齐）：
        - 每个 piece 使用 `id_to_piece` 转为字符串；
        - 以 '▁' 开头表示一个新词的开始，去掉所有前导 '▁' 作为实际文本；
        - 若文本为空（纯空格 token），直接跳过；
        - 非新词的 piece 追加到当前词末尾；
        - 标点符号被视为普通字符，自动附着在当前词末尾（例如 \"moment.\")。
        """
        if not token_ids:
            return []

        words: List[Dict[str, float]] = []

        cur_text: str | None = None
        cur_start_f: int = 0
        cur_end_f: int = 0

        def flush_current():
            nonlocal cur_text, cur_start_f, cur_end_f
            if cur_text is None:
                return
            words.append(
                {
                    "word": cur_text,
                    "start_offset": cur_start_f,
                    "end_offset": cur_end_f,
                    "start": cur_start_f * frame_stride,
                    "end": cur_end_f * frame_stride,
                }
            )
            cur_text = None

        for tid, s_f, d_f in zip(token_ids, token_starts, token_durs):
            piece = self.tokenizer.id_to_piece(tid)
            is_new_word = piece.startswith("▁")
            core = piece.lstrip("▁")

            if core == "":
                # 纯边界标记，不对应实际字符
                continue

            token_start = s_f
            token_end = s_f + d_f

            if cur_text is None or is_new_word:
                # 开启新词（先 flush 旧词）
                flush_current()
                cur_text = core
                cur_start_f = token_start
                cur_end_f = token_end
            else:
                # 同一词的后续子词片段
                cur_text += core
                cur_end_f = token_end

        flush_current()
        return words
