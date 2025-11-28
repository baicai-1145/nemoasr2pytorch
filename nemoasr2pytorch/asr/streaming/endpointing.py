from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class GreedyEndpointing:
    """
    精简版 Greedy endpointing，实现与 NeMo `GreedyEndpointing` 一致的核心行为：
    - 基于 token 序列或时间戳检测句末（EOU）。

    仅保留 RNNT/TDT streaming 需要的几个方法。
    """

    def __init__(
        self,
        vocabulary: List[str],
        ms_per_timestep: int,
        effective_buffer_size_in_secs: Optional[float] = None,
        stop_history_eou: int = -1,
        residue_tokens_at_end: int = 0,
    ) -> None:
        self.vocabulary = vocabulary
        self.ms_per_timestep = ms_per_timestep
        self.sec_per_timestep = ms_per_timestep / 1000.0
        self.stop_history_eou = stop_history_eou
        self.stop_history_eou_ms = stop_history_eou
        self.effective_buffer_size_in_secs = effective_buffer_size_in_secs
        self.residue_tokens_at_end = residue_tokens_at_end

    # 以下两个方法由子类实现，用于判断“静音 token”和“词起始 token”
    def is_token_start_of_word(self, token_id: int) -> bool:  # pragma: no cover - 由子类实现
        raise NotImplementedError

    def is_token_silent(self, token_id: int) -> bool:  # pragma: no cover - 由子类实现
        raise NotImplementedError

    def detect_eou_given_timestamps(
        self,
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
        alignment_length: int,
        stop_history_eou: Optional[int] = None,
    ) -> Tuple[bool, int]:
        """
        参考 NeMo `GreedyEndpointing.detect_eou_given_timestamps` 的实现，
        在时间轴上根据“静音持续长度”与“词边界”检测 EOU。
        """
        if timesteps.numel() == 0 or tokens.numel() == 0:
            return False, -1

        # 取用类上的默认窗口（毫秒），并转换为帧数
        stop_history_eou = stop_history_eou or self.stop_history_eou_ms
        if stop_history_eou is None or stop_history_eou < 0:
            return False, -1
        frames_th = int(stop_history_eou / self.ms_per_timestep)
        if frames_th <= 0:
            return False, -1

        # 简化版逻辑：以最后一个非静音 token 为 pivot，向前查看静音长度
        is_eou = False
        eou_detected_at = -1

        last_timestamp = int(timesteps[-1].item())
        last_token = int(tokens[-1].item())

        # 若末尾已经是静音 token 且持续时间超过阈值，则认为 EOU
        if self.is_token_silent(last_token):
            # 静音起点估计：末尾减去阈值
            silence_start = last_timestamp - frames_th
            if silence_start >= 0:
                is_eou = True
                eou_detected_at = silence_start + frames_th // 2

        # 如果还未检测到，再检查 token 之间是否存在大间隔
        if not is_eou and timesteps.numel() > 1:
            gaps = timesteps[1:] - timesteps[:-1] - 1
            large_gap_mask = gaps > frames_th
            if large_gap_mask.any():
                gap_idx = int(torch.where(large_gap_mask)[0][-1].item())
                is_eou = True
                eou_detected_at = int(timesteps[gap_idx].item() + 1 + frames_th // 2)

        # 保证落在 alignment 范围内
        if is_eou and eou_detected_at >= alignment_length:
            eou_detected_at = alignment_length - 1

        return is_eou, eou_detected_at


class RNNTGreedyEndpointing(GreedyEndpointing):
    """
    针对 RNNT 的 endpointing，主要实现“静音 token”和“词起始 token”的判断。

    BPE 规则：
    - piece 以 '▁' 开头视为词首；
    - 空白/特殊符号（如 <blank>）视为静音。
    """

    def __init__(
        self,
        vocabulary: List[str],
        ms_per_timestep: int,
        effective_buffer_size_in_secs: Optional[float] = None,
        stop_history_eou: int = -1,
        residue_tokens_at_end: int = 0,
    ) -> None:
        super().__init__(
            vocabulary=vocabulary,
            ms_per_timestep=ms_per_timestep,
            effective_buffer_size_in_secs=effective_buffer_size_in_secs,
            stop_history_eou=stop_history_eou,
            residue_tokens_at_end=residue_tokens_at_end,
        )
        self.vocabulary = vocabulary

    def is_token_start_of_word(self, token_id: int) -> bool:
        if token_id < 0 or token_id >= len(self.vocabulary):
            return False
        piece = self.vocabulary[token_id]
        return piece.startswith("▁")

    def is_token_silent(self, token_id: int) -> bool:
        if token_id < 0 or token_id >= len(self.vocabulary):
            return True
        piece = self.vocabulary[token_id]
        # 简单约定：纯空格或特殊 blank 符号视为静音
        return piece.strip() == "" or piece in ("<blank>", "<eps>")

