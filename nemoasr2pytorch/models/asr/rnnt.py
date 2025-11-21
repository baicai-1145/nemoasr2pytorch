from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class RNNTDecoderConfig:
    pred_hidden: int
    pred_rnn_layers: int
    vocab_size: int  # 不含 blank
    dropout: float = 0.2


class PredictionNetwork(nn.Module):
    """简化版 Prediction Net，对齐 NeMo 权重命名：prediction.embed / prediction.dec_rnn.lstm。"""

    def __init__(self, cfg: RNNTDecoderConfig) -> None:
        super().__init__()
        self.blank_idx = cfg.vocab_size
        self.pred_hidden = cfg.pred_hidden
        self.pred_rnn_layers = cfg.pred_rnn_layers

        # +1 for blank_as_pad
        self.embed = nn.Embedding(cfg.vocab_size + 1, cfg.pred_hidden, padding_idx=self.blank_idx)
        lstm = nn.LSTM(
            input_size=cfg.pred_hidden,
            hidden_size=cfg.pred_hidden,
            num_layers=cfg.pred_rnn_layers,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.dec_rnn = nn.Module()
        self.dec_rnn.lstm = lstm

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.pred_rnn_layers, batch_size, self.pred_hidden, device=device, dtype=dtype)
        c0 = torch.zeros(self.pred_rnn_layers, batch_size, self.pred_hidden, device=device, dtype=dtype)
        return h0, c0

    def step(
        self,
        label: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            label: [B] int64
            state: (h, c) each [L, B, H]
        Returns:
            g_t: [B, H]
            new_state: (h, c)
        """
        if label.dim() == 1:
            label = label.unsqueeze(1)  # [B,1]
        emb = self.embed(label)  # [B,1,H]
        out, new_state = self.dec_rnn.lstm(emb, state)  # [B,1,H]
        g = out[:, -1, :]  # [B,H]
        return g, new_state


class RNNTDecoder(nn.Module):
    """封装 PredictionNetwork，使 state_dict 前缀为 decoder.prediction.*。"""

    def __init__(self, cfg: RNNTDecoderConfig) -> None:
        super().__init__()
        self.prediction = PredictionNetwork(cfg)

    @property
    def blank_idx(self) -> int:
        return self.prediction.blank_idx

    @property
    def pred_hidden(self) -> int:
        return self.prediction.pred_hidden

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.prediction.init_state(batch_size, device, dtype)

    def step(
        self,
        label: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.prediction.step(label, state)


@dataclass
class RNNTJointConfig:
    encoder_hidden: int
    pred_hidden: int
    joint_hidden: int
    num_classes: int  # 不含 blank
    num_extra_outputs: int = 0


class RNNTJoint(nn.Module):
    """简化版 Joint Net，保持 NeMo 的 pred/enc/joint_net 命名，并提供与 NeMo 相同的投影接口。"""

    def __init__(self, cfg: RNNTJointConfig) -> None:
        super().__init__()
        self.encoder_hidden = cfg.encoder_hidden
        self.pred_hidden = cfg.pred_hidden
        self.joint_hidden = cfg.joint_hidden

        self._vocab_size = cfg.num_classes
        self._num_extra_outputs = cfg.num_extra_outputs
        self._num_classes = cfg.num_classes + 1 + cfg.num_extra_outputs

        self.pred = nn.Linear(self.pred_hidden, self.joint_hidden)
        self.enc = nn.Linear(self.encoder_hidden, self.joint_hidden)

        activation = nn.ReLU(inplace=True)
        self.joint_net = nn.Sequential(
            activation,
            nn.Dropout(p=0.2),
            nn.Linear(self.joint_hidden, self._num_classes),
        )

    @property
    def num_classes_with_blank(self) -> int:
        return self._num_classes

    @property
    def num_extra_outputs(self) -> int:
        return self._num_extra_outputs

    # --- NeMo 兼容接口：编码器 / 预测网络投影 + joint ---

    def project_encoder(self, f: torch.Tensor) -> torch.Tensor:
        """仅对编码器输出做线性投影，对齐 NeMo `project_encoder`。"""
        return self.enc(f)

    def project_prednet(self, g: torch.Tensor) -> torch.Tensor:
        """仅对预测网络输出做线性投影，对齐 NeMo `project_prednet`。"""
        return self.pred(g)

    def joint_after_projection(self, f_proj: torch.Tensor, g_proj: torch.Tensor) -> torch.Tensor:
        """
        在 encoder / pred 已投影的前提下计算 joint 输出。

        Args:
            f_proj: [B, T, H_joint] 或 [B, 1, H_joint]
            g_proj: [B, U, H_joint] 或 [B, 1, H_joint]
        Returns:
            logits: [B, T, U, V+1+num_extra]
        """
        f = f_proj.unsqueeze(2)  # [B,T,1,H]
        g = g_proj.unsqueeze(1)  # [B,1,U,H]
        inp = f + g
        logits = self.joint_net(inp)
        return logits

    def joint(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        兼容 NeMo 的 joint 接口：先分别投影，再执行 joint_net。

        Args:
            f: [B, T, H_enc]
            g: [B, U, H_pred]
        Returns:
            logits: [B, T, U, V+1+num_extra]
        """
        f_proj = self.project_encoder(f)
        g_proj = self.project_prednet(g)
        return self.joint_after_projection(f_proj, g_proj)
