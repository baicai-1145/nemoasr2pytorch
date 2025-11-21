from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import sentencepiece as spm


@dataclass
class TokenizerConfig:
    model_path: Path
    vocab_path: Path


class TextTokenizer:
    """基于 SentencePiece 的 BPE tokenizer 封装，用于 parakeet-tdt。"""

    def __init__(self, cfg: TokenizerConfig) -> None:
        self.cfg = cfg
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(cfg.model_path))

        # 记录 vocab 大小，方便与 config.labels / joint.num_classes 对比
        self.vocab_size = int(self.sp.vocab_size())

    @classmethod
    def from_nemo_config_dir(cls, model_dir: Path) -> "TextTokenizer":
        """
        从解压后的 NeMo 目录自动发现 tokenizer 相关文件。

        兼容 parakeet-tdt-0.6b-v2 / v3 等不同哈希命名的模型：
        - 优先匹配 *tokenizer*.model，其次任意 *.model；
        - vocab 仅用于调试，优先匹配 *vocab.txt 或 *tokenizer*.vocab。
        """
        # SentencePiece 模型
        model_candidates = sorted(model_dir.glob("*tokenizer*.model"))
        if not model_candidates:
            model_candidates = sorted(model_dir.glob("*.model"))
        if not model_candidates:
            raise FileNotFoundError(f"No SentencePiece .model file found under {model_dir}")
        model_path = model_candidates[0]

        # 词表文件（当前实现未直接使用，保留路径方便调试）
        vocab_candidates = sorted(model_dir.glob("*vocab.txt"))
        if not vocab_candidates:
            vocab_candidates = sorted(model_dir.glob("*tokenizer*.vocab"))
        vocab_path = vocab_candidates[0] if vocab_candidates else model_dir / "vocab.txt"

        cfg = TokenizerConfig(model_path=model_path, vocab_path=vocab_path)
        return cls(cfg)

    def encode(self, text: str) -> List[int]:
        return list(self.sp.encode(text, out_type=int))

    def decode(self, ids: Sequence[int]) -> str:
        return self.sp.decode(ids)

    def piece_to_id(self, piece: str) -> int:
        return int(self.sp.piece_to_id(piece))

    def id_to_piece(self, idx: int) -> str:
        return self.sp.id_to_piece(int(idx))
