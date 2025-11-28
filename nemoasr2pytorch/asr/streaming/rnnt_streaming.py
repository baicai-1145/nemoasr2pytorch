from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from nemoasr2pytorch.models.asr.rnnt import RNNTDecoder, RNNTJoint


@dataclass
class TDTStreamingDecoderState:
    """
    单流 TDT/RNNT 流式解码内部状态。

    仅保存继续解码所需的最小信息：
    - dec_state: 预测网络的隐状态 (h, c)
    - dec_proj: 最近一次非 blank token 之后的 joint 输入投影，供后续 blank label-looping 使用
    - last_token: 最近一次非 blank 预测的 token id（用于上层参考）
    - last_timestamp: 最近一次非 blank token 的全局 encoder 帧 index
    - last_timestamp_lasts: 最近一次 timestamp 连续重复的次数，用于 max_symbols_per_step 防死循环
    """

    dec_state: Tuple[torch.Tensor, torch.Tensor]
    dec_proj: torch.Tensor
    last_token: Optional[int]
    last_timestamp: int
    last_timestamp_lasts: int


class TDTStreamingRNNTDecoder:
    """
    单样本 TDT/RNNT 流式解码器。

    逻辑参考 NeMo 的 TDT label-looping 贪心解码器，但针对本项目的 RNNTDecoder/RNNTJoint
    做了最小封装，只支持 batch_size=1 和单流场景。
    """

    def __init__(
        self,
        decoder: RNNTDecoder,
        joint: RNNTJoint,
        blank_id: int,
        durations: Sequence[int],
        max_symbols_per_step: int = 10,
    ) -> None:
        self.decoder = decoder
        self.joint = joint
        self.blank_id = int(blank_id)
        self.durations = list(int(d) for d in durations)
        self.max_symbols = int(max_symbols_per_step) if max_symbols_per_step is not None else None

    def _init_state(self, device: torch.device, dtype: torch.dtype) -> TDTStreamingDecoderState:
        """
        初始化解码状态：以 blank 作为 <SOS>，跑一步预测网络得到初始 dec_proj。
        """
        dec_state = self.decoder.init_state(batch_size=1, device=device, dtype=dtype)
        labels_tensor = torch.tensor([self.blank_id], device=device, dtype=torch.long)
        g_vec, dec_state = self.decoder.step(labels_tensor, dec_state)
        dec_proj = self.joint.project_prednet(g_vec.unsqueeze(1))[0, 0]  # [H_joint]

        return TDTStreamingDecoderState(
            dec_state=dec_state,
            dec_proj=dec_proj,
            last_token=None,
            last_timestamp=-1,
            last_timestamp_lasts=0,
        )

    @torch.no_grad()
    def decode_chunk(
        self,
        enc_out: torch.Tensor,
        enc_len: torch.Tensor,
        base_time: int,
        state: Optional[TDTStreamingDecoderState] = None,
    ) -> Tuple[List[int], List[int], List[int], TDTStreamingDecoderState]:
        """
        对单个 encoder chunk 做流式解码。

        Args:
            enc_out: [1, D, T_enc] 编码器输出
            enc_len: [1] 有效帧长度
            base_time: 当前 chunk 在整段 encoder 时间轴上的起始帧 index
            state: 上一次解码后的内部状态；None 表示从头开始

        Returns:
            token_ids: 新预测到的 token 序列（可能为空）
            token_starts: 每个 token 的全局起始帧 index
            token_durations: 每个 token 覆盖的帧数
            new_state: 更新后的内部状态，用于下一次解码
        """

        if enc_out.size(0) != 1:
            raise ValueError("TDTStreamingRNNTDecoder 目前仅支持 batch_size=1。")

        device = enc_out.device
        dtype = enc_out.dtype

        # [1, T_enc, D]
        encoder_out = enc_out.transpose(1, 2)
        T_enc = int(enc_len[0].item())
        if T_enc <= 0:
            # 不产生新 token，但仍需确保 state 被初始化
            if state is None:
                state = self._init_state(device=device, dtype=dtype)
            return [], [], [], state

        last_timestep = T_enc - 1
        encoder_proj = self.joint.project_encoder(encoder_out)  # [1, T_enc, H_joint]

        durations_tensor = torch.tensor(self.durations, device=device, dtype=torch.long)
        num_durations = int(durations_tensor.numel())

        # 初始化/继承解码状态
        if state is None:
            state = self._init_state(device=device, dtype=dtype)

        dec_state = state.dec_state
        dec_proj = state.dec_proj
        last_token = state.last_token
        last_timestamp = state.last_timestamp
        last_timestamp_lasts = state.last_timestamp_lasts

        output_tokens: List[int] = []
        token_starts: List[int] = []
        token_durs: List[int] = []

        time_idx = 0  # chunk 内局部 encoder 帧 index
        active = True

        while active:
            active_prev = active

            safe_t = min(time_idx, last_timestep)
            logits = (
                self.joint.joint_after_projection(
                    encoder_proj[:, safe_t : safe_t + 1, :],  # [1,1,H]
                    dec_proj.view(1, 1, -1),  # [1,1,H]
                )
                .squeeze(0)
                .squeeze(0)
                .squeeze(0)
            )  # [V+1+num_durations]

            vocab_plus_blank = logits[:-num_durations]
            duration_logits = logits[-num_durations:]

            label_idx = int(torch.argmax(vocab_plus_blank).item())
            dur_idx = int(torch.argmax(duration_logits).item())
            duration = int(durations_tensor[dur_idx].item())

            is_blank = label_idx == self.blank_id
            if is_blank and duration == 0:
                duration = 1

            label_time_local = time_idx
            time_idx += duration
            active = time_idx < T_enc

            # 在 blank 上继续 label-looping
            while active and is_blank:
                label_time_local = time_idx
                safe_t = min(time_idx, last_timestep)
                logits = (
                    self.joint.joint_after_projection(
                        encoder_proj[:, safe_t : safe_t + 1, :],
                        dec_proj.view(1, 1, -1),
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .squeeze(0)
                )

                vocab_plus_blank = logits[:-num_durations]
                duration_logits = logits[-num_durations:]

                label_idx = int(torch.argmax(vocab_plus_blank).item())
                dur_idx = int(torch.argmax(duration_logits).item())
                duration = int(durations_tensor[dur_idx].item())

                is_blank = label_idx == self.blank_id
                if is_blank and duration == 0:
                    duration = 1

                time_idx += duration
                active = time_idx < T_enc

            # 是否找到非 blank token
            found_label = active_prev and (label_idx != self.blank_id)

            if found_label:
                label_time_global = base_time + label_time_local
                output_tokens.append(label_idx)
                token_starts.append(label_time_global)
                token_durs.append(duration)

                # 维护“同一时间帧发射次数”，防止 max_symbols_per_step 死循环
                if label_time_global == last_timestamp:
                    last_timestamp_lasts += 1
                else:
                    last_timestamp = label_time_global
                    last_timestamp_lasts = 1

                # 更新 decoder 状态与投影后的输出
                new_label = torch.tensor([label_idx], device=device, dtype=torch.long)
                g_vec, dec_state = self.decoder.step(new_label, dec_state)
                dec_proj = self.joint.project_prednet(g_vec.unsqueeze(1))[0, 0]
                last_token = label_idx

            # max_symbols_per_step 防止死循环
            if self.max_symbols is not None and active and label_idx != self.blank_id:
                current_global_time = base_time + time_idx
                if last_timestamp_lasts >= self.max_symbols and last_timestamp == current_global_time:
                    time_idx += 1
                    active = time_idx < T_enc

            if not active:
                break

        new_state = TDTStreamingDecoderState(
            dec_state=dec_state,
            dec_proj=dec_proj,
            last_token=last_token,
            last_timestamp=last_timestamp,
            last_timestamp_lasts=last_timestamp_lasts,
        )

        return output_tokens, token_starts, token_durs, new_state


__all__ = ["TDTStreamingRNNTDecoder", "TDTStreamingDecoderState"]

