from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

INF_VAL = 10000.0


@contextmanager
def avoid_float16_autocast_context():
    """
    NeMo 中用于在某些算子上临时关闭 float16 autocast 的上下文。
    这里实现为简单的 no-op，以保持接口一致且在 FP32 推理下数值等价。
    """
    yield


def _calc_length(
    lengths: torch.Tensor,
    all_paddings: int,
    kernel_size: int,
    stride: int,
    ceil_mode: bool,
    repeat_num: int = 1,
) -> torch.Tensor:
    """Conv/Pool 输出长度计算，与 NeMo 版本保持一致以匹配 subsampling。"""
    add_pad = float(all_paddings - kernel_size)
    one = 1.0
    for _ in range(repeat_num):
        lengths = (lengths.to(torch.float32) + add_pad) / float(stride) + one
        lengths = torch.ceil(lengths) if ceil_mode else torch.floor(lengths)
    return lengths.to(dtype=torch.long)


def _apply_channel_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """在 (B,C,T,F) 的张量上应用时间-频率 mask，mask 形状为 (B,T,F)。"""
    batch_size, channels, time, features = tensor.shape
    expanded_mask = mask.unsqueeze(1).expand(batch_size, channels, time, features)
    return tensor * expanded_mask


def _calculate_conv_output_size(
    input_size: torch.Tensor, kernel_size: int, stride: int, padding: Tuple[int, int]
) -> torch.Tensor:
    """卷积输出长度计算，语义对齐 NeMo `calculate_conv_output_size`。"""
    return (input_size + padding[0] + padding[1] - kernel_size) // stride + 1


class MaskedConvSequential(nn.Sequential):
    """简化版 MaskedConvSequential，用于实现与 NeMo ConvSubsampling 相同的 mask 传播逻辑。"""

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 输入: x [B,T,F]，lengths [B]
        x = x.unsqueeze(1)  # [B,1,T,F]
        current_lengths = lengths.clone().float()
        mask = self._create_mask(x, current_lengths.long())

        for layer in self:
            x = _apply_channel_mask(x, mask)
            x = layer(x)

            if hasattr(layer, "stride") and layer.stride != (1, 1):
                if hasattr(layer, "_left_padding"):
                    padding = (layer._left_padding, layer._right_padding)
                else:
                    padding = layer.padding
                current_lengths = _calculate_conv_output_size(
                    current_lengths, layer.kernel_size[0], layer.stride[0], padding
                )
                mask = self._create_mask(x, current_lengths.long())

        x = _apply_channel_mask(x, mask)
        return x, current_lengths.long()

    def _create_mask(self, tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, channels, time, features = tensor.shape
        time_mask = torch.arange(time, device=tensor.device).expand(batch_size, time) < lengths.unsqueeze(1)
        return time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)


class ConvSubsampling(nn.Module):
    """严格对齐 NeMo `ConvSubsampling` 的 `dw_striding` 分支（不支持 chunking/causal）。"""

    def __init__(
        self,
        subsampling: str,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        conv_channels: int,
    ) -> None:
        super().__init__()
        if subsampling != "dw_striding":
            raise ValueError("ConvSubsampling only supports 'dw_striding' in this implementation.")
        if subsampling_factor % 2 != 0:
            raise ValueError("subsampling_factor must be a power of 2.")

        self._subsampling = subsampling
        self._feat_in = feat_in
        self._feat_out = feat_out
        self._conv_channels = conv_channels

        self._sampling_num = int(math.log(subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False

        self._left_padding = (self._kernel_size - 1) // 2
        self._right_padding = (self._kernel_size - 1) // 2

        layers: List[nn.Module] = []
        in_channels = 1

        # 第一层：普通 Conv2d + ReLU
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=self._left_padding,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        in_channels = conv_channels

        # 后续层：depthwise Conv2d + pointwise Conv2d + ReLU
        for _ in range(self._sampling_num - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=in_channels,
                )
            )
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            in_channels = conv_channels

        # 使用 MaskedConvSequential，实现与 NeMo 相同的 mask 传播
        self.conv = MaskedConvSequential(*layers)

        # 计算单帧输出长度并构建 out 投影
        in_length = torch.tensor(feat_in, dtype=torch.float32)
        out_length = _calc_length(
            lengths=in_length,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        self.out = nn.Linear(conv_channels * int(out_length), feat_out)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, F]
            lengths: [B]
        Returns:
            out: [B, T', feat_out]
            out_lengths: [B]
        """
        # MaskedConvSequential 期望输入 [B,T,F] 和 lengths，[B]
        x, out_lengths = self.conv(x, lengths)  # x: [B, C, T', F']

        B, C, T_out, F_out = x.shape

        x = x.transpose(1, 2).reshape(B, T_out, C * F_out)
        x = self.out(x)
        return x, out_lengths


class PositionalEncoding(nn.Module):
    """
    固定正弦位置编码，接口和实现严格对齐 NeMo `PositionalEncoding`。
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        max_len: int = 5000,
        xscale: Optional[float] = None,
        dropout_rate_emb: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        if dropout_rate_emb > 0:
            self.dropout_emb = nn.Dropout(dropout_rate_emb)
        else:
            self.dropout_emb = None

    def create_pe(self, positions: torch.Tensor, dtype: torch.dtype) -> None:
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(INF_VAL) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        if hasattr(self, "pe"):
            self.pe = pe
        else:
            self.register_buffer("pe", pe, persistent=False)

    def extend_pe(self, length: int, device: torch.device, dtype: torch.dtype) -> None:
        """当需要时扩展位置编码（与 NeMo 逻辑一致）。"""
        if hasattr(self, "pe") and self.pe.size(1) >= length:
            return
        positions = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x: torch.Tensor, cache_len: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D]
            cache_len: 用于流式缓存的位置偏移
        Returns:
            x_pe: [B, T, D]
            pos_emb: [1, T+cache_len, D]
        """
        input_len = x.size(1) + cache_len
        if self.xscale:
            x = x * self.xscale
        pos_emb = self.pe[:, :input_len]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        x = x + pos_emb[:, cache_len:]
        return self.dropout(x), pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """
    相对位置编码，实现严格对齐 NeMo `RelPositionalEncoding`。
    """

    def extend_pe(self, length: int, device: torch.device, dtype: torch.dtype) -> None:
        """根据长度重置并扩展相对位置编码。"""
        needed_size = 2 * length - 1
        if hasattr(self, "pe") and self.pe.size(1) >= needed_size:
            return
        # 位置从 (L-1) 到 -(L-1)
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x: torch.Tensor, cache_len: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        与 NeMo 一致：返回 dropout(x) 与截取好的 pos_emb（不将 pos_emb 加到 x 上）。
        """
        if self.xscale:
            x = x * self.xscale

        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb


class MultiHeadAttention(nn.Module):
    """
    完整端口 NeMo 的 MultiHeadAttention，实现 scaled dot-product attention。
    仅在 use_pytorch_sdpa=False 情况下会走手写路径，与 NeMo 数值行为一致。
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        max_cache_len: int = 0,
        use_bias: bool = True,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.use_pytorch_sdpa = use_pytorch_sdpa
        if self.use_pytorch_sdpa and use_pytorch_sdpa_backends:
            # 将字符串 backend 名转换为 PyTorch 的 SDPBackend 枚举
            if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "SDPBackend"):
                use_pytorch_sdpa_backends = list(
                    map(lambda name: getattr(torch.nn.attention.SDPBackend, name), use_pytorch_sdpa_backends)
                )
            else:
                use_pytorch_sdpa_backends = None
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends

        self.cache_drop_size: Optional[int] = None
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        assert n_feat % n_head == 0
        # d_v 始终等于 d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout_rate)

        self._max_cache_len = max_cache_len

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将输入映射到 q/k/v 并拆成多头。
        Args:
            query: [B, T1, D]
            key  : [B, T2, D]
            value: [B, T2, D]
        Returns:
            q, k, v: [B, H, T*, d_k]
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        根据打分和 mask 计算注意力输出（与 NeMo 完全一致）。
        Args:
            value : [B, H, T2, d_k]
            scores: [B, H, T1, T2]
            mask  : [B, T1, T2] 或 None（True 表示需要 mask）
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B,1,T1,T2]
            scores = scores.masked_fill(mask, -INF_VAL)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # [B,H,T1,d_k]
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # [B,T1,D]
        return self.linear_out(x)

    def update_cache(
        self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor, cache: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        与 NeMo 一致的 cache 更新逻辑；当前推理路径不使用 cache，但保持接口。
        """
        if cache is not None:
            key = value = torch.cat([cache, key], dim=1)
            q_keep_size = query.shape[1] - self.cache_drop_size
            cache = torch.cat([cache[:, q_keep_size:, :], query[:, :q_keep_size, :]], dim=1)
        return key, value, query, cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
        pos_emb: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ):
        """
        标准 scaled dot-product attention。
        Args:
            query: [B, T1, D]
            key  : [B, T2, D]
            value: [B, T2, D]
            mask : [B, T1, T2]（True 表示需要 mask）或 None
            cache: [B, T_cache, D] 或 None
        Returns:
            out 或 (out, cache)
        """
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)

            if self.use_pytorch_sdpa:
                n_batch = value.size(0)
                if mask is not None:
                    # 对于 sdpa，mask=True 表示保留，因此需要取反
                    mask = ~mask.unsqueeze(1)  # [B,1,T1,T2]

                dropout_rate = self.dropout_rate if self.training else 0.0
                if self.use_pytorch_sdpa_backends:
                    with torch.nn.attention.sdpa_kernel(self.use_pytorch_sdpa_backends):
                        out = torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, attn_mask=mask, dropout_p=dropout_rate
                        )
                else:
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=mask, dropout_p=dropout_rate
                    )

                if mask is not None:
                    all_masked_rows = torch.all(~mask, dim=-1)
                    all_masked_rows.unsqueeze_(-1)
                    out = out.masked_fill(all_masked_rows, 0.0)

                out = out.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)
                out = self.linear_out(out)
            else:
                scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
                out = self.forward_attention(v, scores, mask)

        if cache is None:
            return out
        else:
            return out, cache


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """
    完整端口 NeMo 的 RelPositionMultiHeadAttention（Transformer-XL 相对位置注意力）。
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        pos_bias_u: Optional[torch.Tensor],
        pos_bias_v: Optional[torch.Tensor],
        max_cache_len: int = 0,
        use_bias: bool = True,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            dropout_rate=dropout_rate,
            max_cache_len=max_cache_len,
            use_bias=use_bias,
            use_pytorch_sdpa=use_pytorch_sdpa,
            use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
        )
        # 位置编码线性变换
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # 用于相对位置打分的两个可学习偏置
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        相对位置移位操作。
        Args:
            x: [B, H, T1, 2*T1-1]
        """
        b, h, qlen, pos_len = x.size()
        x = torch.nn.functional.pad(x, pad=(1, 0))  # [B,H,T1,2*T1]
        x = x.view(b, h, -1, qlen)  # [B,H,2*T1, T1]
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # [B,H,T1,2*T1-1]
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
        pos_emb: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ):
        """
        带相对位置编码的 scaled dot-product attention。
        Args:
            query: [B, T1, D]
            key  : [B, T2, D]
            value: [B, T2, D]
            mask : [B, T1, T2] 或 None
            pos_emb: [1, T1+T_cache*2-1, D]
        """
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            q = q.transpose(1, 2)  # [B, T1, H, d_k]

            n_batch_pos = pos_emb.size(0)
            n_batch = value.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # [B_pos, H, T1, d_k]

            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)  # [B,H,T1,d_k]
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

            # 位置部分打分
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))  # [B,H,T1,2*T1-1]
            matrix_bd = self.rel_shift(matrix_bd)

            if self.use_pytorch_sdpa:
                scale_factor = 1.0 / math.sqrt(q_with_bias_u.size(-1))
                matrix_bd = matrix_bd[:, :, :, : k.size(-2)] * scale_factor

                if mask is not None:
                    mask = mask.unsqueeze(1)
                    matrix_bd.masked_fill_(mask, -INF_VAL)

                dropout_rate = self.dropout_rate if self.training else 0.0
                if self.use_pytorch_sdpa_backends:
                    with torch.nn.attention.sdpa_kernel(self.use_pytorch_sdpa_backends):
                        out = torch.nn.functional.scaled_dot_product_attention(
                            q_with_bias_u, k, v, attn_mask=matrix_bd, dropout_p=dropout_rate
                        )
                else:
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q_with_bias_u, k, v, attn_mask=matrix_bd, dropout_p=dropout_rate
                    )

                if mask is not None:
                    all_masked_rows = torch.all(mask, dim=-1)
                    all_masked_rows.unsqueeze_(-1)
                    all_masked_rows = all_masked_rows.expand(-1, out.size(1), -1, out.size(-1))
                    out = out.masked_fill(all_masked_rows, 0.0)

                out = out.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)
                out = self.linear_out(out)
            else:
                matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
                matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
                scores = (matrix_ac + matrix_bd) / self.s_d_k
                out = self.forward_attention(v, scores, mask)

        if cache is None:
            return out
        else:
            return out, cache


class ConformerFeedForward(nn.Module):
    """
    基于 NeMo ConformerFeedForward 的精简实现（使用 SiLU/Swish 激活）。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float, use_bias: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class CausalConv1D(nn.Conv1d):
    """
    复制 NeMo CausalConv1D 的核心行为（仅保留非流式推理需要的逻辑）：
    - 支持 `padding=None` 或 `padding=[left, right]` 两种模式；
    - forward 时先按 left/right 进行 F.pad，再调用父类 conv。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[Sequence[int]] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        if padding is None:
            left_padding = kernel_size - 1
            right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                left_padding = padding
                right_padding = padding
            elif (
                isinstance(padding, (list, tuple))
                and len(padding) == 2
                and padding[0] + padding[1] == kernel_size - 1
            ):
                left_padding = padding[0]
                right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._left_padding = left_padding
        self._right_padding = right_padding

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None):
        # 只实现非流式路径：忽略 cache，按固定左右 padding 处理
        x = F.pad(x, pad=(self._left_padding, self._right_padding))
        x = super().forward(x)
        if cache is None:
            return x
        else:
            # 为保持签名兼容，简单返回 (x, cache)
            return x, cache


class ConformerConvolution(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        norm_type: str = "batch_norm",
        conv_context_size: Optional[Sequence[int]] = None,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        if conv_context_size is None:
            conv_context_size = ((kernel_size - 1) // 2, (kernel_size - 1) // 2)

        self.d_model = d_model
        self.kernel_size = kernel_size

        pointwise_out = d_model * 2
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=pointwise_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )

        dw_in = d_model
        padding = conv_context_size
        self.depthwise_conv = CausalConv1D(
            in_channels=dw_in,
            out_channels=dw_in,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dw_in,
            bias=use_bias,
        )

        if norm_type == "batch_norm":
            self.batch_norm = nn.BatchNorm1d(dw_in)
        else:
            raise ValueError(f"Unsupported conv_norm_type: {norm_type}")

        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_in,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, cache: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, T, D]
        """
        x = x.transpose(1, 2)  # [B,D,T]
        x = self.pointwise_conv1(x)
        x = torch.nn.functional.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return x


class ConformerLayer(nn.Module):
    """单个 Conformer block，接口/子模块命名与 NeMo 对齐。"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        self_attention_model: str,
        n_heads: int,
        conv_kernel_size: int,
        conv_norm_type: str,
        conv_context_size: Sequence[int],
        dropout: float,
        dropout_att: float,
        pos_bias_u: Optional[torch.Tensor],
        pos_bias_v: Optional[torch.Tensor],
        att_context_size: Sequence[int],
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.self_attention_model = self_attention_model
        self.fc_factor = 0.5

        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
        )

        self.norm_self_att = nn.LayerNorm(d_model)
        if self_attention_model == "rel_pos":
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                use_bias=use_bias,
            )
        elif self_attention_model == "abs_pos":
            self.self_attn = MultiHeadAttention(
                n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att, use_bias=use_bias
            )
        else:
            raise ValueError(f"Unsupported self_attention_model: {self_attention_model}")

        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        cache_last_channel: Optional[torch.Tensor] = None,
        cache_last_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        """
        # FFN1
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        # Self-attention
        x = self.norm_self_att(residual)
        if self.self_attention_model == "rel_pos":
            if pos_emb is None:
                raise ValueError("pos_emb must be provided for rel_pos attention.")
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb, cache=None)
        else:
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=None, cache=None)

        residual = residual + self.dropout(x)

        # Conv
        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask, cache=None)
        residual = residual + self.dropout(x)

        # FFN2
        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)
        return x


@dataclass
class ConformerEncoderConfig:
    feat_in: int
    n_layers: int
    d_model: int
    feat_out: int = -1
    subsampling: str = "dw_striding"
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    ff_expansion_factor: int = 4
    self_attention_model: str = "rel_pos"
    n_heads: int = 8
    att_context_size: Optional[Sequence[int]] = None
    att_context_style: str = "regular"
    xscaling: bool = False
    untie_biases: bool = True
    pos_emb_max_len: int = 5000
    conv_kernel_size: int = 9
    conv_norm_type: str = "batch_norm"
    conv_context_size: Optional[Sequence[int]] = None
    dropout: float = 0.1
    dropout_pre_encoder: float = 0.1
    dropout_emb: float = 0.0
    dropout_att: float = 0.1
    use_bias: bool = False


class ConformerEncoder(nn.Module):
    """简化版 ConformerEncoder，保留与 NeMo 对齐的权重命名，用于加载 parakeet 权重。"""

    def __init__(self, cfg: ConformerEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        d_model = cfg.d_model
        d_ff = d_model * cfg.ff_expansion_factor

        self._feat_in = cfg.feat_in
        self.d_model = d_model
        self.n_layers = cfg.n_layers

        # subsampling
        if cfg.subsampling_conv_channels == -1:
            conv_channels = d_model
        else:
            conv_channels = cfg.subsampling_conv_channels

        if cfg.subsampling and cfg.subsampling_factor > 1:
            self.pre_encode = ConvSubsampling(
                subsampling=cfg.subsampling,
                subsampling_factor=cfg.subsampling_factor,
                feat_in=cfg.feat_in,
                feat_out=d_model,
                conv_channels=conv_channels,
            )
        else:
            self.pre_encode = nn.Linear(cfg.feat_in, d_model)

        # position encoding
        self.pos_emb_max_len = cfg.pos_emb_max_len
        xscale = math.sqrt(d_model) if cfg.xscaling else None
        if cfg.self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=cfg.dropout_pre_encoder,
                max_len=cfg.pos_emb_max_len,
                xscale=xscale,
                dropout_rate_emb=cfg.dropout_emb,
            )
        elif cfg.self_attention_model == "abs_pos":
            self.pos_enc = PositionalEncoding(
                d_model=d_model,
                dropout_rate=cfg.dropout_pre_encoder,
                max_len=cfg.pos_emb_max_len,
                xscale=xscale,
                dropout_rate_emb=cfg.dropout_emb,
            )
        else:
            raise ValueError(f"Unsupported self_attention_model: {cfg.self_attention_model}")

        # att context / conv context 简化处理
        if cfg.att_context_size is None:
            att_context_size = [-1, -1]
        else:
            att_context_size = list(cfg.att_context_size)
        if cfg.conv_context_size is None:
            conv_context_size = [(cfg.conv_kernel_size - 1) // 2, (cfg.conv_kernel_size - 1) // 2]
        else:
            conv_context_size = list(cfg.conv_context_size)

        # pos biases（只在 untie_biases=False 时使用）
        if not cfg.untie_biases and cfg.self_attention_model == "rel_pos":
            d_head = d_model // cfg.n_heads
            pos_bias_u = nn.Parameter(torch.zeros(cfg.n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.zeros(cfg.n_heads, d_head))
        else:
            pos_bias_u = None
            pos_bias_v = None

        # Conformer layers
        self.layers = nn.ModuleList()
        for _ in range(cfg.n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=cfg.self_attention_model,
                n_heads=cfg.n_heads,
                conv_kernel_size=cfg.conv_kernel_size,
                conv_norm_type=cfg.conv_norm_type,
                conv_context_size=conv_context_size,
                dropout=cfg.dropout,
                dropout_att=cfg.dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                att_context_size=att_context_size,
                use_bias=cfg.use_bias,
            )
            self.layers.append(layer)

        if cfg.feat_out > 0 and cfg.feat_out != d_model:
            self.out_proj = nn.Linear(d_model, cfg.feat_out)
            self._feat_out = cfg.feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model

    @property
    def feat_out(self) -> int:
        return self._feat_out

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "ConformerEncoder":
        # 严格参考 NeMo ConformerEncoder 的配置字段构造本地 Config
        att_context_size = cfg.get("att_context_size")
        if att_context_size is None:
            att_context_size = [-1, -1]

        conv_context_size = cfg.get("conv_context_size")
        conv_kernel_size = cfg.get("conv_kernel_size", 9)
        if conv_context_size is None:
            # 与 NeMo _calc_context_sizes 中 conv_context_size=None 分支保持一致
            conv_context_size = [(conv_kernel_size - 1) // 2, (conv_kernel_size - 1) // 2]
        elif isinstance(conv_context_size, str) and conv_context_size == "causal":
            conv_context_size = [conv_kernel_size - 1, 0]

        return cls(
            ConformerEncoderConfig(
                feat_in=cfg["feat_in"],
                feat_out=cfg.get("feat_out", -1),
                n_layers=cfg["n_layers"],
                d_model=cfg["d_model"],
                use_bias=cfg.get("use_bias", False),
                subsampling=cfg.get("subsampling", "dw_striding"),
                subsampling_factor=cfg.get("subsampling_factor", 8),
                subsampling_conv_channels=cfg.get("subsampling_conv_channels", -1),
                ff_expansion_factor=cfg.get("ff_expansion_factor", 4),
                self_attention_model=cfg.get("self_attention_model", "rel_pos"),
                n_heads=cfg.get("n_heads", 8),
                att_context_size=att_context_size,
                att_context_style=cfg.get("att_context_style", "regular"),
                xscaling=cfg.get("xscaling", False),
                untie_biases=cfg.get("untie_biases", True),
                pos_emb_max_len=cfg.get("pos_emb_max_len", 5000),
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=cfg.get("conv_norm_type", "batch_norm"),
                conv_context_size=conv_context_size,
                dropout=cfg.get("dropout", 0.1),
                dropout_pre_encoder=cfg.get("dropout_pre_encoder", 0.1),
                dropout_emb=cfg.get("dropout_emb", 0.0),
                dropout_att=cfg.get("dropout_att", 0.1),
            )
        )

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_signal: [B, F, T]
            length: [B] （帧数）
        Returns:
            encoded: [B, D, T']
            encoded_lengths: [B]
        """
        x = audio_signal.transpose(1, 2)  # [B, T, F]

        if isinstance(self.pre_encode, ConvSubsampling):
            x, length = self.pre_encode(x, length)
        else:
            x = self.pre_encode(x)

        # 位置编码：确保相对位置编码的 pe 长度充足且为相对坐标
        self.pos_enc.extend_pe(length=x.size(1), device=x.device, dtype=x.dtype)
        x, pos_emb = self.pos_enc(x, cache_len=0)

        B, T = x.size(0), x.size(1)
        device = x.device

        # 构造与 NeMo _create_masks 等价的 pad_mask / att_mask（仅支持 att_context_style='regular' 且 rel_pos）
        # pad_mask: True 表示 padding 位置
        time_ids = torch.arange(0, T, device=device).unsqueeze(0).expand(B, -1)  # [B,T]
        pad_mask = time_ids >= length.unsqueeze(1)  # [B,T]

        att_mask = None
        if self.cfg.att_context_style == "regular":
            # 基础全可见矩阵，后续按 context 裁剪
            base = torch.ones(1, T, T, dtype=torch.bool, device=device)
            left_ctx, right_ctx = self.cfg.att_context_size or [-1, -1]
            if left_ctx >= 0:
                base = base.triu(diagonal=-left_ctx)
            if right_ctx >= 0:
                base = base.tril(diagonal=right_ctx)

            # 将 padding 位置屏蔽掉
            pad_mask_for_att = ~pad_mask  # True 表示有效
            pad_mask_for_att = pad_mask_for_att.unsqueeze(1) & pad_mask_for_att.unsqueeze(2)  # [B,T,T]
            att_mask = base[:, :T, :T] & pad_mask_for_att  # True 表示可见
            att_mask = ~att_mask  # True 表示需要 mask

        for layer in self.layers:
            x = layer(x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)

        if self.out_proj is not None:
            x = self.out_proj(x)

        x = x.transpose(1, 2)  # [B, D, T']
        return x, length
