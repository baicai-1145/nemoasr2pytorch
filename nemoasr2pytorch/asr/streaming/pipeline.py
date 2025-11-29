from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from nemoasr2pytorch.asr.streaming.endpointing import RNNTGreedyEndpointing
from nemoasr2pytorch.asr.streaming.greedy_decoder import ClippedRNNTGreedyDecoder
from nemoasr2pytorch.asr.streaming.state import ParakeetStreamingState
from nemoasr2pytorch.models.asr.parakeet_tdt import ParakeetTDTModel


@dataclass
class StreamingConfig:
    """精简版 Streaming 配置，包含与 NeMo `streaming` 小节类似的关键参数。"""

    sample_rate: int
    chunk_size: float = 8.0  # 每个 chunk 的时长（秒）
    left_padding_size: float = 0.0
    right_padding_size: float = 0.0
    stop_history_eou_ms: int = -1  # 静音触发 EOU 的窗口（毫秒），-1 表示禁用


class BufferedParakeetPipeline:
    """
    剪裁版 buffered RNNT/TDT 流式流水线，仅支持单流、单文件场景。

    设计参考 `BufferedRNNTPipeline`，但去掉多流和 cache-aware 能力。
    """

    def __init__(self, model: ParakeetTDTModel, cfg: StreamingConfig, device: torch.device | str | None = None):
        self.model = model
        self.cfg = cfg

        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)

        self.sample_rate = cfg.sample_rate
        self.chunk_size = cfg.chunk_size
        self.left_padding_size = cfg.left_padding_size
        self.right_padding_size = cfg.right_padding_size

        # 与 NeMo 一致：buffer_size = chunk + left + right
        self.buffer_size_in_secs = self.chunk_size + self.left_padding_size + self.right_padding_size

        # Mel 帧步长 & encoder 步长
        self.window_stride = model.frame_stride
        self.model_stride_in_secs = model.model_stride_in_secs

        # 期望的特征 buffer 长度（帧数）
        self.expected_feature_buffer_len = int(self.buffer_size_in_secs / self.window_stride)

        # tokens-per-frame 相关参数（按 BufferedRNNTPipeline 公式）
        self.mid_delay = int(
            torch.ceil(torch.tensor((self.chunk_size + self.right_padding_size) / self.model_stride_in_secs)).item()
        )
        self.tokens_per_frame_float = self.chunk_size / self.model_stride_in_secs
        self.tokens_per_left_padding_float = self.left_padding_size / self.model_stride_in_secs
        self.tokens_per_right_padding_float = self.right_padding_size / self.model_stride_in_secs
        self.tokens_per_frame = int(torch.ceil(torch.tensor(self.tokens_per_frame_float)).item())
        self.tokens_per_left_padding = int(torch.ceil(torch.tensor(self.tokens_per_left_padding_float)).item())
        self.tokens_per_right_padding = int(torch.ceil(torch.tensor(self.tokens_per_right_padding_float)).item())

        if self.left_padding_size or self.right_padding_size:
            self.initial_delay = self.right_padding_size / self.model_stride_in_secs
        else:
            self.initial_delay = 0.0

        # 构建 BPE 词表列表，用于 endpointing / greedy decoder
        self.vocabulary: List[str] = [self.model.tokenizer.id_to_piece(i) for i in range(self.model.tokenizer.vocab_size)]

        # Endpointing：基于 token 序列和时间戳检测 EOU
        self.model_stride_in_milliseconds = int(round(self.model_stride_in_secs * 1000.0))
        self.endpointer = RNNTGreedyEndpointing(
            vocabulary=self.vocabulary,
            ms_per_timestep=self.model_stride_in_milliseconds,
            effective_buffer_size_in_secs=self.buffer_size_in_secs,
            stop_history_eou=cfg.stop_history_eou_ms,
            residue_tokens_at_end=0,
        )
        self.greedy_rnnt_decoder = ClippedRNNTGreedyDecoder(
            vocabulary=self.vocabulary,
            tokens_per_frame=self.tokens_per_frame,
            conf_func=None,
            endpointer=self.endpointer,
        )

        # 用于填充的零编码
        self.zero_encoded = self._init_zero_encoded()

        # 流式状态与累积 token 序列（可用于逐步推理）
        self.state = ParakeetStreamingState()
        self._all_token_ids: List[int] = []
        # 累积的波形前缀，用于在每个 step 上对「截至目前的整段音频」做一次完整 TDT 解码，
        # 保证流式文本在任何时刻都尽可能接近离线 NeMo 结果。
        self._waveform_prefix: torch.Tensor | None = None
        self._last_word_offsets: List[dict] = []

    def _init_zero_encoded(self) -> torch.Tensor:
        """
        构造一个全零 encoder 输出，用于处理 padding roll 后的空洞。
        """
        buffer_size_in_samples = int(self.buffer_size_in_secs * self.sample_rate)
        zero_buffer = torch.zeros(1, buffer_size_in_samples, device=self.device)
        # 复用 ParakeetTDTModel.encode，自动对齐 encoder dtype / 逻辑
        enc_out, _ = self.model.encode(zero_buffer)
        # [B, D, T_enc] -> 取 batch=0
        return enc_out[0]

    def reset_state(self) -> None:
        """重置流式状态，准备开始新的一段音频。"""
        self.state.reset()
        self.state.set_global_offset(-self.initial_delay)
        self._all_token_ids.clear()
        self._waveform_prefix = None
        self._last_word_offsets = []

    @property
    def last_word_offsets(self) -> List[dict]:
        """
        返回最近一次解码得到的 word-level 时间戳信息（如果有）。
        形如：
            {
                "word": "moment.",
                "start_offset": 103,
                "end_offset": 109,
                "start": 8.24,
                "end": 8.72,
            }
        """
        return self._last_word_offsets

    def trim_prefix_seconds(self, seconds: float) -> None:
        """
        在内部累积的 waveform 前缀上裁剪掉最前面的若干秒音频，用于在上层
        依据句级边界裁剪上下文，避免前缀无限增长导致推理时间越来越长。
        """
        if self._waveform_prefix is None:
            return
        if seconds <= 0.0:
            return
        samples = int(seconds * self.sample_rate)
        if samples <= 0:
            return
        if samples >= self._waveform_prefix.numel():
            # 全部裁掉，后续再追加新 chunk
            self._waveform_prefix = None
        else:
            self._waveform_prefix = self._waveform_prefix[samples:]

    def _waveform_to_chunks(self, waveform: torch.Tensor) -> List[Tuple[int, int]]:
        """
        将整段 waveform 分成多个 chunk（单位：样本点 index）。
        返回 [(start_sample, end_sample), ...]。
        """
        total_len = waveform.shape[-1]
        chunk_samples = int(self.chunk_size * self.sample_rate)
        if chunk_samples <= 0:
            return [(0, total_len)]

        chunks: List[Tuple[int, int]] = []
        start = 0
        while start < total_len:
            end = min(start + chunk_samples, total_len)
            chunks.append((start, end))
            start = end
        return chunks

    @torch.no_grad()
    def encode_raw_chunk(self, chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对单个音频 chunk 做特征提取 + encoder 编码。
        返回：
            enc_out: [1, D, T_enc]
            enc_len: [1]
        """
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        B, T = chunk.shape
        lengths = torch.full((B,), T, dtype=torch.long, device=self.device)
        feats, feat_len = self.model.preprocessor(chunk.to(self.device), lengths)

        enc_dtype = next(self.model.encoder.parameters()).dtype
        feats = feats.to(dtype=enc_dtype)

        enc_out, enc_len = self.model.encoder(feats, feat_len)
        return enc_out, enc_len

    @torch.no_grad()
    def stream_step(self, chunk_waveform: torch.Tensor, is_last: bool = False, is_first: bool = False) -> str:
        """
        流式推理的单步：
        - 输入一个 1D chunk waveform；
        - 更新内部 StreamingState 和累积 token 列表；
        - 返回当前为止的完整转写文本（方便调试/打印）。
        """
        if chunk_waveform.dim() == 2:
            chunk_waveform = chunk_waveform.squeeze(0)
        chunk_waveform = chunk_waveform.to(self.device)

        # 累积前缀波形：每次在「截至当前为止的整段音频」上做一次完整 TDT 解码，
        # 确保流式文本在语义上尽量与离线 NeMo 结果一致（代价是计算量更大）。
        if self._waveform_prefix is None:
            self._waveform_prefix = chunk_waveform
        else:
            self._waveform_prefix = torch.cat([self._waveform_prefix, chunk_waveform], dim=0)

        full_waveform = self._waveform_prefix

        # 直接复用 Parakeet 的 encode + TDT 解码（与离线路径完全一致）
        enc_out, enc_len = self.model.encode(full_waveform)
        decode_res = self.model._tdt_decoder.decode_with_timestamps(enc_out, enc_len)

        token_ids = list(decode_res.token_ids)
        timesteps = list(decode_res.token_starts)
        token_durs = list(decode_res.token_durations)

        # 保存当前 step 的 word-level 时间戳，供上层按句子边界裁剪上下文使用
        self._last_word_offsets = self.model._bpe_tokens_to_word_offsets(token_ids, timesteps, token_durs)

        # 用完整前缀的 token/time 序列更新内部状态
        self.state.tokens = token_ids
        self.state.timesteps = timesteps
        self.state.confidences = [0.0] * len(token_ids)
        self.state.current_step_tokens = token_ids

        # 不在这里做 EOU 清空，完整前缀的文本全部作为当前 partial
        self._all_token_ids = token_ids
        if token_ids:
            self.state.partial_transcript = self.model.tokenizer.decode(token_ids)
        else:
            self.state.partial_transcript = ""

        return self.state.partial_transcript

    @torch.no_grad()
    def run_streaming(self, waveform: torch.Tensor) -> str:
        """
        对单条 waveform（[T]，采样率 cfg.sample_rate）执行简化版 streaming 推理。
        返回整段文本。
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        waveform = waveform.to(self.device)

        self.reset_state()
        chunks = self._waveform_to_chunks(waveform)

        for chunk_idx, (s, e) in enumerate(chunks):
            is_last_chunk = chunk_idx == len(chunks) - 1
            chunk_wave = waveform[s:e]
            self.stream_step(chunk_wave, is_last=is_last_chunk, is_first=(chunk_idx == 0))

        # 返回最终文本：直接基于累积 token 序列解码
        if self._all_token_ids:
            return self.model.tokenizer.decode(self._all_token_ids)
        return ""
