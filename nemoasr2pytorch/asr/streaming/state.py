from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .state_utils import CLOSE_IN_TIME_TH, OVERLAP_SEARCH_TH, detect_overlap, merge_timesteps


@dataclass
class ParakeetStreamingState:
    """
    精简版流式状态，仅保留与单流 RNNT/TDT 解码相关的核心字段。

    设计遵循 NeMo `StreamingState` / `RNNTStreamingState` 的语义，但去掉了
    多流、ITN、复杂分段等能力，便于在本项目内使用。
    """

    # 相对整段音频的时间偏移（单位：encoder 帧 index 或秒的外层统一管理）
    global_offset: float = 0.0

    # 累积的 token / timestep / 置信度（自上次 EOU 之后）
    tokens: List[int] = field(default_factory=list)
    timesteps: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    # 当前 step 的新增 token（便于上层构造 partial transcript）
    current_step_tokens: List[int] = field(default_factory=list)

    # 上一次输出的 token 及其时间 index，用于重叠检测
    last_token: Optional[int] = None
    last_token_idx: Optional[int] = None

    # EOU 检测相关
    eou_detected_before: bool = False
    decoder_start_idx: int = 0
    decoder_end_idx: int = 0

    # 记录每次检测到 EOU 时的全局时间步（encoder 帧 index）
    eou_timesteps: List[int] = field(default_factory=list)

    # RNNT 解码器内部状态（例如隐状态等），具体类型由解码器维护
    hyp_decoding_state: Any = None

    # 文本级状态：final_transcript 表示已确认的部分；partial_transcript 表示当前句子的临时结果
    final_transcript: str = ""
    partial_transcript: str = ""

    def reset(self) -> None:
        """重置所有状态。"""
        self.global_offset = 0.0
        self.tokens.clear()
        self.timesteps.clear()
        self.confidences.clear()
        self.current_step_tokens.clear()
        self.last_token = None
        self.last_token_idx = None
        self.eou_detected_before = False
        self.decoder_start_idx = 0
        self.decoder_end_idx = 0
        self.eou_timesteps.clear()
        self.hyp_decoding_state = None
        self.final_transcript = ""
        self.partial_transcript = ""

    # ---- 与 NeMo 语义一致的若干辅助方法 ----

    def set_global_offset(self, start_offset: float) -> None:
        self.global_offset = float(start_offset)

    def increment_global_offset(self, shift: float) -> None:
        self.global_offset += float(shift)

    def set_last_token(self, token: Optional[int], idx: Optional[int]) -> None:
        if token is None or idx is None:
            self.last_token = None
            self.last_token_idx = None
        else:
            self.last_token = token
            # 这里直接存储“绝对”时间步，外层使用时可按需乘以 model_stride_in_secs
            self.last_token_idx = int(idx + self.global_offset)

    def update_from_decoder_results(self, start_idx: int, end_idx: int) -> None:
        """记录当前 clip 在对齐上的起止 index，便于下一次调用 clipped 解码器。"""
        self.decoder_start_idx = int(start_idx)
        self.decoder_end_idx = int(end_idx)

    # ---- 与 NeMo 类似的 token/timestep 合并逻辑 ----

    def _update_state_tokens(self, output: Dict, skip: int) -> None:
        """
        参考 NeMo `_update_state`：在可选跳过前几个 token 的情况下，合并新旧 token 和 timesteps。
        """
        current_tokens = list(output["tokens"])
        current_timesteps = list(output["timesteps"])
        current_confidences = list(output.get("confidences", [0.0] * len(current_tokens)))

        if skip > 0:
            current_tokens = current_tokens[skip:]
            current_timesteps = current_timesteps[skip:]
            current_confidences = current_confidences[skip:]

        self.current_step_tokens = list(current_tokens)
        self.tokens.extend(current_tokens)
        self.confidences.extend(current_confidences)
        self.timesteps = merge_timesteps(self.timesteps, current_timesteps)

        # 更新 last_token / last_token_idx
        if self.tokens and self.timesteps:
            self.set_last_token(self.tokens[-1], self.timesteps[-1])
        else:
            self.last_token = None
            self.last_token_idx = None

    def update_state(self, completed_output: Dict, eou_detected: bool) -> None:
        """
        参考 NeMo `StreamingState.update_state`，基于 overlap 检测合并 token/timestep。
        """
        tokens = list(completed_output.get("tokens", []))
        if not tokens:
            self.last_token = None
            self.last_token_idx = None
            return

        timesteps = list(completed_output.get("timesteps", []))
        confidences = list(completed_output.get("confidences", [0.0] * len(tokens)))

        overlap = 0
        if not self.eou_detected_before:
            overlap = detect_overlap(
                state_tokens=self.tokens,
                state_timesteps=self.timesteps,
                new_tokens=tokens,
                new_timesteps=timesteps,
                overlap_search_th=OVERLAP_SEARCH_TH,
                close_in_time_th=CLOSE_IN_TIME_TH,
            )

        if (
            self.eou_detected_before
            and self.last_token == tokens[0]
            and self.last_token_idx is not None
            and abs(self.last_token_idx - timesteps[0]) <= CLOSE_IN_TIME_TH
        ):
            overlap = max(overlap, 1)

        self._update_state_tokens(
            {"tokens": tokens, "timesteps": timesteps, "confidences": confidences},
            skip=overlap,
        )
        self.eou_detected_before = eou_detected

    def cleanup_after_eou(self) -> None:
        """
        在检测到 EOU 后清空当前句子的 token 状态，保留 final_transcript。
        """

        self.tokens.clear()
        self.timesteps.clear()
        self.confidences.clear()
        self.current_step_tokens.clear()
        self.last_token = None
        self.last_token_idx = None

    # ---- 状态的轻量序列化（不包含解码器内部张量）----

    def to_dict(self) -> dict:
        """
        导出可 JSON 序列化的状态快照（不包含 hyp_decoding_state）。

        主要用于未来跨会话保存/恢复全局时间轴与 token 序列，
        解码器内部张量状态仍建议在新会话中重新初始化。
        """
        return {
            "global_offset": self.global_offset,
            "tokens": list(self.tokens),
            "timesteps": list(self.timesteps),
            "confidences": list(self.confidences),
            "last_token": self.last_token,
            "last_token_idx": self.last_token_idx,
            "eou_detected_before": self.eou_detected_before,
            "decoder_start_idx": self.decoder_start_idx,
            "decoder_end_idx": self.decoder_end_idx,
            "eou_timesteps": list(self.eou_timesteps),
            "final_transcript": self.final_transcript,
            "partial_transcript": self.partial_transcript,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParakeetStreamingState":
        """
        根据 to_dict() 的输出构造新的状态对象，hyp_decoding_state 置为 None。
        """
        state = cls()
        state.global_offset = float(data.get("global_offset", 0.0))
        state.tokens = list(data.get("tokens", []))
        state.timesteps = list(data.get("timesteps", []))
        state.confidences = list(data.get("confidences", []))
        state.last_token = data.get("last_token", None)
        state.last_token_idx = data.get("last_token_idx", None)
        state.eou_detected_before = bool(data.get("eou_detected_before", False))
        state.decoder_start_idx = int(data.get("decoder_start_idx", 0))
        state.decoder_end_idx = int(data.get("decoder_end_idx", 0))
        state.eou_timesteps = list(data.get("eou_timesteps", []))
        state.final_transcript = str(data.get("final_transcript", ""))
        state.partial_transcript = str(data.get("partial_transcript", ""))
        state.hyp_decoding_state = None
        return state
