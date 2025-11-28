from __future__ import annotations

"""
从 NeMo 迁移的部分 state 管理工具：
- merge_timesteps: 合并时间戳序列并保持单调
- detect_overlap / find_max_overlap: 检测旧 token 尾部与新 token 头部的重叠，用于尾部重写
- merge_segment_tail / merge_word_tail: 句段/词级尾部合并（为将来扩展预留）
"""

from typing import Callable, List, Tuple

from .constants import POST_WORD_PUNCTUATION, PRE_WORD_PUNCTUATION
from .text_segment import TextSegment, Word

CLOSE_IN_TIME_TH = 2.0
OVERLAP_SEARCH_TH = 3


def merge_timesteps(timesteps1: List[float], timesteps2: List[float]) -> List[float]:
    """
    参考 NeMo：合并两段时间戳序列，保持整体递增。
    """
    if not timesteps1 and not timesteps2:
        return []

    if timesteps1:
        first = timesteps1[0]
        if first < 0:
            for i, t in enumerate(timesteps1):
                timesteps1[i] = t - first

    if timesteps2:
        first = timesteps2[0]
        if first < 0:
            for i, t in enumerate(timesteps2):
                timesteps2[i] = t - first

    if not timesteps1:
        return timesteps2
    if not timesteps2:
        return timesteps1

    gap = timesteps2[0] - timesteps1[-1]
    if gap <= 0:
        return timesteps1 + [t + abs(gap) + 1 for t in timesteps2]
    return timesteps1 + timesteps2


def find_max_overlap(state_tokens: List[int], new_tokens: List[int], limit: int) -> int:
    """
    在 state_tokens 后缀与 new_tokens 前缀之间寻找最大重叠长度（不超过 limit）。
    """
    max_overlap = 0
    max_k = min(len(state_tokens), len(new_tokens), limit)
    for k in range(1, max_k + 1):
        if state_tokens[-k:] == new_tokens[:k]:
            max_overlap = k
    return max_overlap


def detect_overlap(
    state_tokens: List[int],
    state_timesteps: List[float],
    new_tokens: List[int],
    new_timesteps: List[float],
    overlap_search_th: int = OVERLAP_SEARCH_TH,
    close_in_time_th: float = CLOSE_IN_TIME_TH,
) -> int:
    """
    检测 state_tokens 尾部与 new_tokens 头部之间的重叠长度（NeMo 同款逻辑）。
    """
    overlap = 0
    if state_tokens:
        overlap = find_max_overlap(state_tokens, new_tokens, overlap_search_th)
        if overlap > 0:
            close_in_time = (new_timesteps[overlap - 1] - state_timesteps[-overlap]) <= close_in_time_th
            overlap = overlap if close_in_time else 0
    return overlap


def merge_segment_tail(
    segment_head: TextSegment,
    segment_tail: TextSegment,
    conf_aggregator: Callable | None = None,
) -> TextSegment:
    """
    将 segment_tail 合并到 segment_head（NeMo 同款逻辑的精简实现）。
    """
    head = segment_head.copy()

    # 去掉末尾标点后再拼接
    if head.text and (last_char := head.text[-1]) and last_char in POST_WORD_PUNCTUATION:
        head.text = head.text.rstrip(last_char)

    head.text += segment_tail.text
    head.end = segment_tail.end

    if conf_aggregator is not None:
        head.conf = conf_aggregator([head.conf, segment_tail.conf])

    return head


def merge_word_tail(
    word_head: Word,
    word_tail: Word,
    pnc_word_head: Word | None = None,
    conf_aggregator: Callable | None = None,
) -> Tuple[Word, Word | None]:
    """
    将 word_tail 合并到 word_head，并按需要维护带标点/大小写的 pnc_word_head。
    """
    head = word_head.copy()
    head_text = head.text

    if head_text and (last_char := head_text[-1]) and last_char in POST_WORD_PUNCTUATION:
        head.text = head_text.rstrip(last_char)

    head.text += word_tail.text
    head.end = word_tail.end

    if conf_aggregator is not None:
        head.conf = conf_aggregator([head.conf, word_tail.conf])

    pnc_head: Word | None = None
    if pnc_word_head is not None:
        last_char = pnc_word_head.text[-1] if pnc_word_head.text else None
        first_char = pnc_word_head.text[0] if pnc_word_head.text else None

        pnc_head = head.copy()

        if last_char in POST_WORD_PUNCTUATION:
            if pnc_head.text and pnc_head.text[-1] not in POST_WORD_PUNCTUATION:
                pnc_head.text = pnc_head.text + last_char

        if first_char in PRE_WORD_PUNCTUATION:
            if pnc_head.text and pnc_head.text[0] not in PRE_WORD_PUNCTUATION:
                pnc_head.text = first_char + pnc_head.text

        if first_char and first_char.isupper():
            pnc_head.capitalize()

    return head, pnc_head

