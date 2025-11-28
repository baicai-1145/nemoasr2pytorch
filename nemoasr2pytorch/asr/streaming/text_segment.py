from __future__ import annotations

"""
从 NeMo 迁移的 TextSegment / Word 结构体，用于句段/词级文本管理。
"""

from functools import lru_cache
from typing import List

from .constants import DEFAULT_SEMIOTIC_CLASS, SEP_REPLACEABLE_PUNCTUATION


@lru_cache(maxsize=5)
def get_translation_table(punct_marks_frozen: frozenset[str], sep: str) -> dict:
    replace_map = {mark: sep if mark in SEP_REPLACEABLE_PUNCTUATION else "" for mark in punct_marks_frozen}
    return str.maketrans(replace_map)


def normalize_text(text: str, punct_marks: set[str], sep: str) -> str:
    trans_table = get_translation_table(frozenset(punct_marks), sep)
    return text.translate(trans_table).lower()


def validate_init_params(
    text: str, start: float, end: float, conf: float, semiotic_class: str | None = None, strict: bool = False
) -> None:
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    if not isinstance(start, (int, float)):
        raise TypeError(f"start must be numeric, got {type(start).__name__}")
    if not isinstance(end, (int, float)):
        raise TypeError(f"end must be numeric, got {type(end).__name__}")
    if not isinstance(conf, (int, float)):
        raise TypeError(f"conf must be numeric, got {type(conf).__name__}")

    if semiotic_class is not None and not isinstance(semiotic_class, str):
        raise TypeError(f"semiotic_class must be a string, got {type(semiotic_class).__name__}")

    if strict:
        if start >= end:
            raise ValueError(f"start time ({start}) must be less than end time ({end})")
        if conf < 0 or conf > 1:
            raise ValueError(f"confidence ({conf}) must be between 0 and 1")


class TextSegment:
    """单个连续文本片段（带起止时间与置信度）。"""

    __slots__ = ["_text", "_start", "_end", "_conf"]

    def __init__(self, text: str, start: float, end: float, conf: float) -> None:
        validate_init_params(text, start, end, conf, strict=True)
        self._text = text
        self._start = start
        self._end = end
        self._conf = conf

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"text must be a string, got {type(value).__name__}")
        self._text = value

    @property
    def start(self) -> float:
        return self._start

    @start.setter
    def start(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"start time must be numeric, got {type(value).__name__}")
        self._start = value

    @property
    def end(self) -> float:
        return self._end

    @end.setter
    def end(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"end time must be numeric, got {type(value).__name__}")
        self._end = value

    @property
    def conf(self) -> float:
        return self._conf

    @conf.setter
    def conf(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"conf must be numeric, got {type(value).__name__}")
        if value < 0 or value > 1:
            raise ValueError(f"confidence ({value}) must be between 0 and 1")
        self._conf = value

    @property
    def duration(self) -> float:
        return self._end - self._start

    def copy(self) -> "TextSegment":
        return TextSegment(text=self.text, start=self.start, end=self.end, conf=self.conf)

    def capitalize(self) -> None:
        self._text = self._text.capitalize()

    def with_normalized_text(self, punct_marks: set[str], sep: str = "") -> "TextSegment":
        obj_copy = self.copy()
        obj_copy._text = normalize_text(self._text, punct_marks, sep)
        return obj_copy

    def normalize_text_inplace(self, punct_marks: set[str], sep: str = "") -> None:
        self._text = normalize_text(self._text, punct_marks, sep)

    def to_dict(self) -> dict:
        return {"text": self.text, "start": self.start, "end": self.end, "conf": self.conf}


class Word(TextSegment):
    """单词级片段，附带 semiotic_class。"""

    __slots__ = ["_semiotic_class"]

    def __init__(
        self,
        text: str,
        start: float,
        end: float,
        conf: float,
        semiotic_class: str = DEFAULT_SEMIOTIC_CLASS,
    ) -> None:
        validate_init_params(text, start, end, conf, semiotic_class, strict=True)
        super().__init__(text, start, end, conf)
        self._semiotic_class = semiotic_class

    @property
    def semiotic_class(self) -> str:
        return self._semiotic_class

    @semiotic_class.setter
    def semiotic_class(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"semiotic_class must be a string, got {type(value).__name__}")
        self._semiotic_class = value

    def copy(self) -> "Word":
        return Word(text=self.text, start=self.start, end=self.end, conf=self.conf, semiotic_class=self.semiotic_class)

    def to_dict(self) -> dict:
        return super().to_dict() | {"semiotic_class": self.semiotic_class}


def join_segments(segments: List[List[TextSegment]], sep: str) -> List[str]:
    return [sep.join([s.text for s in items]) for items in segments]


def normalize_segments_inplace(
    segments: List[TextSegment] | List[List[TextSegment]],
    punct_marks: set[str],
    sep: str = " ",
) -> None:
    for item in segments:
        if isinstance(item, list):
            for segment in item:
                segment.normalize_text_inplace(punct_marks, sep)
        elif isinstance(item, TextSegment):
            segment = item
            segment.normalize_text_inplace(punct_marks, sep)
        else:
            raise ValueError(f"Invalid item type: {type(item)}. Expected `TextSegment` or `List[TextSegment]`.")


