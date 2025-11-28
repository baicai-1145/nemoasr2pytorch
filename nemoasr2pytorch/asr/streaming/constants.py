from __future__ import annotations

"""
流式 ASR 相关的基础常量。

本文件对应 NeMo `nemo/collections/asr/inference/utils/constants.py`
中的一小部分，与文本拼接和 SentencePiece 相关的标点常量。
"""

# Punctuation related constants
POST_WORD_PUNCTUATION = set(".,?")
PRE_WORD_PUNCTUATION = set("¿")
SEP_REPLACEABLE_PUNCTUATION = set("-_")
SENTENCEPIECE_UNDERSCORE = "▁"

# ITN related constants（保留语义，与 NeMo 对齐）
DEFAULT_SEMIOTIC_CLASS = "name"

