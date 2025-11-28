from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from nemoasr2pytorch.asr.streaming.endpointing import RNNTGreedyEndpointing


@dataclass
class ClippedDecodeOutput:
    tokens: List[int]
    timesteps: List[int]
    confidences: List[float]
    last_token: Optional[int]
    last_token_idx: Optional[int]


class ClippedRNNTGreedyDecoder:
    """
    精简版 Clipped RNNT Greedy decoder。

    作用：
    - 在给定对齐范围 [clip_start, clip_end) 内，从全局 token/timestep 序列中裁剪出本次有效的子序列；
    - 可选地使用 endpointing（静音 / VAD）做 EOU 检测；
    - 维护 decoder 的 state_start_idx / state_end_idx，供上层 StreamingState 使用。
    """

    def __init__(
        self,
        vocabulary: List[str],
        tokens_per_frame: int,
        conf_func: Optional[Callable[[List[int]], List[float]]] = None,
        endpointer: Optional[RNNTGreedyEndpointing] = None,
    ) -> None:
        self.vocabulary = vocabulary
        self.tokens_per_frame = int(tokens_per_frame)
        self.conf_func = conf_func
        self.endpointer = endpointer

    @staticmethod
    def _extract_clipped_and_tail(
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
        start_idx: int,
        end_idx: int,
        return_tail_result: bool,
    ) -> Tuple[List[int], List[int], List[int]]:
        if timesteps.numel() == 0:
            return [], [], []
        mask_clip = (timesteps >= start_idx) & (timesteps < end_idx)
        clipped_timesteps = timesteps[mask_clip].tolist()
        clipped_tokens = tokens[mask_clip].tolist()

        tail_tokens: List[int] = []
        if return_tail_result:
            mask_tail = timesteps >= end_idx
            if mask_tail.any():
                tail_tokens = tokens[mask_tail].tolist()
        return clipped_timesteps, clipped_tokens, tail_tokens

    def __call__(
        self,
        global_timesteps: torch.Tensor,
        tokens: torch.Tensor,
        clip_start: int,
        clip_end: int,
        alignment_length: int,
        is_last: bool = True,
        is_start: bool = True,
        return_tail_result: bool = False,
        state_start_idx: int = 0,
        state_end_idx: int = 0,
        timestamp_offset: int = 0,
        vad_segments: Optional[torch.Tensor] = None,
        stop_history_eou: Optional[int] = None,
    ) -> Tuple[Dict, Dict, bool, int, int]:
        """
        参考 NeMo `ClippedRNNTGreedyDecoder.__call__` 的核心行为：
        - 根据 clip 区间裁剪 token/timestep；
        - 调用 endpointing 检测 EOU；
        - 返回 clipped_output, tail_output, is_eou, new_start_idx, new_end_idx。
        """
        if global_timesteps.numel() == 0 or tokens.numel() == 0:
            empty = {"tokens": [], "timesteps": [], "confidences": [], "last_token": None, "last_token_idx": None}
            return empty, {"tokens": []}, True, state_start_idx, state_end_idx

        if timestamp_offset:
            timesteps = global_timesteps - timestamp_offset
        else:
            timesteps = global_timesteps

        is_eou = is_last
        eou_detected_at = alignment_length

        start_idx, end_idx = state_start_idx, state_end_idx
        if end_idx > clip_start:
            end_idx -= self.tokens_per_frame
            start_idx = end_idx
        if is_start:
            start_idx, end_idx = clip_start, clip_start
        elif end_idx <= clip_start:
            start_idx, end_idx = clip_start, clip_end

        # endpointing：基于时间戳检测 EOU
        if not is_eou and self.endpointer is not None:
            if vad_segments is not None and vad_segments.numel() > 0:
                # 为简化：当前暂不使用 VAD 段落做 EOU，保留接口
                pass
            else:
                is_eou, eou_detected_at = self.endpointer.detect_eou_given_timestamps(
                    timesteps=timesteps, tokens=tokens, alignment_length=alignment_length, stop_history_eou=stop_history_eou
                )

        if is_eou and eou_detected_at > end_idx:
            end_idx = min(eou_detected_at, alignment_length)

        if clip_start <= end_idx < clip_end:
            end_idx = clip_end
            is_eou = False

        clipped_timesteps, clipped_tokens, tail_tokens = self._extract_clipped_and_tail(
            timesteps, tokens, start_idx, end_idx, return_tail_result
        )

        if timestamp_offset:
            clipped_timesteps = [t + timestamp_offset for t in clipped_timesteps]

        clipped_output: Dict = {
            "tokens": clipped_tokens,
            "timesteps": clipped_timesteps,
            "confidences": [0.0] * len(clipped_tokens) if clipped_tokens else [],
            "last_token": None,
            "last_token_idx": None,
        }
        if clipped_tokens:
            clipped_output["last_token"] = clipped_tokens[-1]
            clipped_output["last_token_idx"] = clipped_timesteps[-1] if clipped_timesteps else None

        tail_output: Dict = {"tokens": tail_tokens}
        return clipped_output, tail_output, is_eou, start_idx, end_idx

