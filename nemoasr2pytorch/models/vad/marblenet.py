from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


def _get_same_padding(kernel_size: int, stride: int, dilation: int) -> int:
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (dilation * (kernel_size - 1)) // 2


@dataclass
class JasperBlockConfig:
    filters: int
    repeat: int
    kernel: Sequence[int]
    stride: Sequence[int]
    dilation: Sequence[int]
    dropout: float
    residual: bool
    separable: bool = False


class SeparableConvUnit(nn.Module):
    """单个 depthwise-separable conv + BN + ReLU + Dropout 单元."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
        separable: bool,
    ) -> None:
        super().__init__()
        padding = _get_same_padding(kernel_size, stride, dilation)

        if separable:
            self.depthwise = nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            )
            self.pointwise = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        else:
            self.depthwise = None
            self.pointwise = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            )

        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.depthwise is not None:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MarbleNetBlock(nn.Module):
    """简化版 MarbleNet block，由若干个 SeparableConvUnit 组成，可选 residual。"""

    def __init__(self, in_channels: int, cfg: JasperBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()

        current_in = in_channels
        for i in range(cfg.repeat):
            unit = SeparableConvUnit(
                in_channels=current_in,
                out_channels=cfg.filters,
                kernel_size=cfg.kernel[0],
                stride=cfg.stride[0],
                dilation=cfg.dilation[0],
                dropout=cfg.dropout,
                separable=cfg.separable,
            )
            self.layers.append(unit)
            current_in = cfg.filters

        self.out_channels = cfg.filters

        if cfg.residual:
            # residual 1x1 conv + BN
            self.residual_conv = nn.Conv1d(
                in_channels,
                cfg.filters,
                kernel_size=1,
                stride=cfg.stride[0],
                padding=0,
                bias=False,
            )
            self.residual_bn = nn.BatchNorm1d(cfg.filters, eps=1e-3, momentum=0.1)
        else:
            self.residual_conv = None
            self.residual_bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for layer in self.layers:
            x = layer(x)

        if self.residual_conv is not None and self.residual_bn is not None:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)
            x = x + residual

        return x


class MarbleNetEncoder(nn.Module):
    """基于 Jasper 配置构建的 MarbleNet 编码器，仅用于推理。"""

    def __init__(self, feat_in: int, jasper_cfg: List[dict]) -> None:
        super().__init__()
        blocks: List[MarbleNetBlock] = []
        in_ch = feat_in
        for block_cfg in jasper_cfg:
            cfg = JasperBlockConfig(
                filters=block_cfg["filters"],
                repeat=block_cfg["repeat"],
                kernel=block_cfg["kernel"],
                stride=block_cfg["stride"],
                dilation=block_cfg["dilation"],
                dropout=block_cfg["dropout"],
                residual=block_cfg["residual"],
                separable=block_cfg.get("separable", False),
            )
            block = MarbleNetBlock(in_channels=in_ch, cfg=cfg)
            blocks.append(block)
            in_ch = cfg.filters

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = in_ch

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, F, T]
        Returns:
            encoded: [B, C, T']
            encoded_lengths: [B] (这里直接使用 T'，不严格跟踪卷积长度变化)
        """
        x = features
        for block in self.blocks:
            x = block(x)

        lengths = torch.full(
            size=(x.size(0),),
            fill_value=x.size(2),
            dtype=torch.long,
            device=x.device,
        )
        return x, lengths

