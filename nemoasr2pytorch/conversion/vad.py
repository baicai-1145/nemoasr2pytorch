from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import yaml

from nemoasr2pytorch.audio.melspectrogram import AudioToMelSpectrogramPreprocessor
from nemoasr2pytorch.models.vad.frame_vad_model import FrameVADDecoder, FrameVADModel
from nemoasr2pytorch.models.vad.marblenet import MarbleNetEncoder, MarbleNetBlock, SeparableConvUnit


def _load_preprocessor_weights(preprocessor: AudioToMelSpectrogramPreprocessor, state_dict: Dict[str, torch.Tensor]) -> None:
    featurizer = preprocessor.featurizer
    window_key = "preprocessor.featurizer.window"
    fb_key = "preprocessor.featurizer.fb"
    with torch.no_grad():
        if window_key in state_dict and featurizer.window is not None:
            featurizer.window.copy_(state_dict[window_key])
        if fb_key in state_dict and featurizer.fb is not None:
            featurizer.fb.copy_(state_dict[fb_key])


def _load_marblenet_block_weights(
    block: MarbleNetBlock,
    block_idx: int,
    block_cfg: dict,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    """将 NeMo encoder.encoder.{i}.* 权重映射到 MarbleNetBlock."""

    prefix = f"encoder.encoder.{block_idx}."
    repeat = block_cfg["repeat"]
    separable = block_cfg.get("separable", False)

    def get(key: str) -> torch.Tensor:
        return state_dict[prefix + key]

    with torch.no_grad():
        if repeat == 1:
            unit = block.layers[0]
            if separable:
                # depthwise + pointwise + BN
                unit.depthwise.weight.copy_(get("mconv.0.conv.weight"))
                unit.pointwise.weight.copy_(get("mconv.1.conv.weight"))
                unit.bn.weight.copy_(get("mconv.2.weight"))
                unit.bn.bias.copy_(get("mconv.2.bias"))
                unit.bn.running_mean.copy_(get("mconv.2.running_mean"))
                unit.bn.running_var.copy_(get("mconv.2.running_var"))
            else:
                unit.pointwise.weight.copy_(get("mconv.0.conv.weight"))
                unit.bn.weight.copy_(get("mconv.1.weight"))
                unit.bn.bias.copy_(get("mconv.1.bias"))
                unit.bn.running_mean.copy_(get("mconv.1.running_mean"))
                unit.bn.running_var.copy_(get("mconv.1.running_var"))
        elif repeat == 2 and separable:
            # NeMo: mconv.0/1/2 + act/drop + mconv.5/6/7
            for r, base in enumerate((0, 5)):
                unit = block.layers[r]
                unit.depthwise.weight.copy_(get(f"mconv.{base}.conv.weight"))
                unit.pointwise.weight.copy_(get(f"mconv.{base+1}.conv.weight"))
                unit.bn.weight.copy_(get(f"mconv.{base+2}.weight"))
                unit.bn.bias.copy_(get(f"mconv.{base+2}.bias"))
                unit.bn.running_mean.copy_(get(f"mconv.{base+2}.running_mean"))
                unit.bn.running_var.copy_(get(f"mconv.{base+2}.running_var"))
        else:
            raise NotImplementedError(f"Unsupported MarbleNet block config: repeat={repeat}, separable={separable}")

        # residual 分支（如果存在）
        if block_cfg.get("residual", False):
            block.residual_conv.weight.copy_(get("res.0.0.conv.weight"))
            block.residual_bn.weight.copy_(get("res.0.1.weight"))
            block.residual_bn.bias.copy_(get("res.0.1.bias"))
            block.residual_bn.running_mean.copy_(get("res.0.1.running_mean"))
            block.residual_bn.running_var.copy_(get("res.0.1.running_var"))


def _load_encoder_weights(
    encoder: MarbleNetEncoder,
    encoder_cfg: dict,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    jasper_cfg: List[dict] = encoder_cfg["jasper"]
    assert len(jasper_cfg) == len(encoder.blocks)
    for idx, cfg in enumerate(jasper_cfg):
        _load_marblenet_block_weights(encoder.blocks[idx], idx, cfg, state_dict)


def _load_decoder_weights(decoder: FrameVADDecoder, state_dict: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        decoder.linear.weight.copy_(state_dict["decoder.layer0.weight"])
        decoder.linear.bias.copy_(state_dict["decoder.layer0.bias"])


def load_frame_vad_from_nemo(
    config_path: str | Path,
    ckpt_path: str | Path,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> FrameVADModel:
    """从解压后的 .nemo 配置和 ckpt 构建 FrameVADModel。"""

    config_path = Path(config_path)
    ckpt_path = Path(ckpt_path)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    labels = cfg.get("labels", ["0", "1"])

    # 1) 预处理器
    preproc_cfg = cfg["preprocessor"]
    preprocessor = AudioToMelSpectrogramPreprocessor.from_config(preproc_cfg)

    # 2) 编码器
    encoder_cfg = cfg["encoder"]
    feat_in = encoder_cfg["feat_in"]
    jasper_cfg = encoder_cfg["jasper"]
    encoder = MarbleNetEncoder(feat_in=feat_in, jasper_cfg=jasper_cfg)

    # 3) 解码器（线性层）
    decoder_cfg = cfg["decoder"]
    num_classes = decoder_cfg.get("num_classes", 2)
    decoder = FrameVADDecoder(in_features=encoder.out_channels, num_classes=num_classes)

    model = FrameVADModel(preprocessor=preprocessor, encoder=encoder, decoder=decoder, labels=labels)

    # 加载权重
    obj = torch.load(ckpt_path, map_location="cpu")
    state_dict: Dict[str, torch.Tensor] = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

    _load_preprocessor_weights(preprocessor, state_dict)
    _load_encoder_weights(encoder, encoder_cfg, state_dict)
    _load_decoder_weights(decoder, state_dict)

    if device is None:
        device = torch.device("cpu")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model

