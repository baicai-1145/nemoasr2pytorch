from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from nemoasr2pytorch.models.asr.rnnt import RNNTDecoder, RNNTJoint


@dataclass
class TDTGreedyConfig:
    blank_id: int
    durations: Sequence[int]
    max_symbols_per_step: int = 10


class GreedyTDTDecoder:
    """TDT label-looping 贪心解码（单样本版本）。

    按 NeMo `GreedyBatchedTDTLabelLoopingComputer.torch_impl` 的状态机语义实现 label-looping，
    但仅支持 batch_size=1，且不实现对齐/置信度等附加功能。
    """

    def __init__(self, decoder: RNNTDecoder, joint: RNNTJoint, cfg: TDTGreedyConfig) -> None:
        self.decoder = decoder
        self.joint = joint
        self.cfg = cfg

    @torch.no_grad()
    def decode(
        self,
        enc_out: torch.Tensor,
        enc_len: torch.Tensor,
    ) -> List[int]:
        """
        Args:
            enc_out: [1, D, T] 编码器输出
            enc_len: [1] 有效帧长度
        Returns:
            token_ids: List[int]（BPE id 序列）
        """

        if enc_out.size(0) != 1:
            raise ValueError("GreedyTDTDecoder 目前仅支持 batch_size=1。")

        device = enc_out.device
        dtype = enc_out.dtype

        # [1, T, D]
        encoder_out = enc_out.transpose(1, 2)
        T_enc = int(enc_len[0].item())
        if T_enc <= 0:
            return []

        last_timestep = T_enc - 1

        # 将 encoder 输出投影到 joint 隐空间，对齐 NeMo `project_encoder`
        encoder_proj = self.joint.project_encoder(encoder_out)  # [1,T,H]

        # durations 及相关参数
        durations_tensor = torch.tensor(list(self.cfg.durations), device=device, dtype=torch.long)
        num_durations = int(durations_tensor.numel())
        blank_id = self.cfg.blank_id
        max_symbols = self.cfg.max_symbols_per_step

        # 初始 time index / 活跃标记
        time_idx = 0
        active = True

        # decoder 初始状态：以 blank 作为 <SOS>
        state = self.decoder.init_state(batch_size=1, device=device, dtype=dtype)
        labels_tensor = torch.tensor([blank_id], device=device, dtype=torch.long)
        g_vec, state = self.decoder.step(labels_tensor, state)  # [1,H_pred]
        dec_proj = self.joint.project_prednet(g_vec.unsqueeze(1))[0, 0]  # [H_joint]

        # 记录输出 token 及“同一时间帧发射次数”
        output_tokens: List[int] = []
        last_timestamp = -1
        last_timestamp_lasts = 0

        while active:
            active_prev = active

            # ---------- Stage 1.1：当前时间位置第一次 joint ----------
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

            is_blank = label_idx == blank_id
            if is_blank and duration == 0:
                duration = 1

            label_time = time_idx
            time_idx += duration
            active = time_idx < T_enc

            # ---------- Stage 1.2：在 blank 上继续 label-looping ----------
            while active and is_blank:
                label_time = time_idx
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

                is_blank = label_idx == blank_id
                if is_blank and duration == 0:
                    duration = 1

                time_idx += duration
                active = time_idx < T_enc

            # ---------- Stage 2：是否找到非 blank token ----------
            found_label = active_prev and (label_idx != blank_id)

            if found_label:
                output_tokens.append(label_idx)

                # 维护与 NeMo BatchedHyps 等价的 last_timestamp / last_timestamp_lasts
                if label_time == last_timestamp:
                    last_timestamp_lasts += 1
                else:
                    last_timestamp = label_time
                    last_timestamp_lasts = 1

                # 更新 decoder 状态与投影后的输出
                new_label = torch.tensor([label_idx], device=device, dtype=torch.long)
                g_vec, state = self.decoder.step(new_label, state)
                dec_proj = self.joint.project_prednet(g_vec.unsqueeze(1))[0, 0]

            # ---------- Stage 3：max_symbols_per_step 防止死循环 ----------
            if max_symbols is not None and active and label_idx != blank_id:
                if last_timestamp_lasts >= max_symbols and last_timestamp == time_idx:
                    time_idx += 1
                    active = time_idx < T_enc

            if not active:
                break

        return output_tokens


__all__ = ["TDTGreedyConfig", "GreedyTDTDecoder"]
