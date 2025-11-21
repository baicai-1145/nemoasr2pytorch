from __future__ import annotations

from pathlib import Path

import torch
import yaml

from nemoasr2pytorch.audio.melspectrogram import AudioToMelSpectrogramPreprocessor
from nemoasr2pytorch.models.asr.conformer import ConformerEncoder
from nemoasr2pytorch.models.asr.parakeet_tdt import ParakeetTDTConfig, ParakeetTDTModel
from nemoasr2pytorch.models.asr.rnnt import RNNTDecoder, RNNTDecoderConfig, RNNTJoint, RNNTJointConfig
from nemoasr2pytorch.text.tokenizer import TextTokenizer


def load_parakeet_tdt_from_nemo(
    config_path: str | Path,
    ckpt_path: str | Path,
    tokenizer_dir: str | Path,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> ParakeetTDTModel:
    """从解压好的 parakeet-tdt 配置和 ckpt 构建 ParakeetTDTModel。"""

    config_path = Path(config_path)
    ckpt_path = Path(ckpt_path)
    tokenizer_dir = Path(tokenizer_dir)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sample_rate = cfg["sample_rate"]
    preproc_cfg = cfg["preprocessor"]
    encoder_cfg = cfg["encoder"]
    decoder_cfg = cfg["decoder"]
    joint_cfg = cfg["joint"]
    decoding_cfg = cfg["decoding"]
    loss_cfg = cfg["loss"]
    labels = cfg["labels"]

    durations = decoding_cfg.get("durations", [])

    # 1) 预处理器
    preprocessor = AudioToMelSpectrogramPreprocessor.from_config(preproc_cfg)

    # 2) 编码器（Conformer）
    encoder = ConformerEncoder.from_config_dict(encoder_cfg)

    # 3) Decoder & Joint
    vocab_size = len(labels)
    rd_cfg = RNNTDecoderConfig(
        pred_hidden=decoder_cfg["prednet"]["pred_hidden"],
        pred_rnn_layers=decoder_cfg["prednet"]["pred_rnn_layers"],
        vocab_size=vocab_size,
        dropout=decoder_cfg["prednet"].get("dropout", 0.2),
    )
    decoder = RNNTDecoder(rd_cfg)

    rj_cfg = RNNTJointConfig(
        encoder_hidden=joint_cfg["jointnet"]["encoder_hidden"],
        pred_hidden=joint_cfg["jointnet"]["pred_hidden"],
        joint_hidden=joint_cfg["jointnet"]["joint_hidden"],
        num_classes=joint_cfg["num_classes"],
        num_extra_outputs=joint_cfg.get("num_extra_outputs", 0),
    )
    joint = RNNTJoint(rj_cfg)

    # 4) tokenizer
    tokenizer = TextTokenizer.from_nemo_config_dir(tokenizer_dir)

    # 5) wrap model
    blank_id = vocab_size  # 与 NeMo TDT 一致：blank 索引 = vocab_size
    model_cfg = ParakeetTDTConfig(sample_rate=sample_rate, blank_id=blank_id, durations=durations)
    model = ParakeetTDTModel(
        preprocessor=preprocessor,
        encoder=encoder,
        decoder=decoder,
        joint=joint,
        tokenizer=tokenizer,
        cfg=model_cfg,
    )

    # 6) 加载权重
    obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

    # 因为我们刻意保持模块命名与 NeMo 一致，大部分权重可以直接 load_state_dict。
    # 已验证 strict=False 时 missing/unexpected 为空。
    model.load_state_dict(state_dict, strict=False)

    if device is None:
        device = torch.device("cpu")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model
