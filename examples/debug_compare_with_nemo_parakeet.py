from __future__ import annotations

"""
对比 NeMo 原始 parakeet-tdt-0.6b-v2 与本项目 ParakeetTDTModel 的中间结果：
- Mel 特征 (preprocessor 输出)
- Conformer encoder 输出

使用说明：
- 需要本仓库根目录下存在 NeMo 源码目录 `NeMo/`，且 NeMo 依赖已安装（你之前能在 WSL 跑 NeMo 就说明 OK）。
- 需要 `parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo` 文件存在。
"""

from pathlib import Path
import sys

import torch

# 允许从 examples/ 目录直接运行
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 将 NeMo 源码加入路径（假设本仓库下有 NeMo/）
NEMO_ROOT = REPO_ROOT / "NeMo"
if NEMO_ROOT.is_dir() and str(NEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(NEMO_ROOT))


def _load_nemo_parakeet(device: torch.device):
    try:
        from nemo.collections.asr.models import EncDecRNNTBPEModel  # type: ignore
    except Exception as e:  # pragma: no cover - 仅在本地调试时使用
        print("导入 NeMo 失败，请确认已正确安装 NeMo，且 `NeMo/` 在 PYTHONPATH 中。")
        print(f"Import error: {e}")
        sys.exit(1)

    nemo_path = REPO_ROOT / "parakeet-tdt-0.6b-v2" / "parakeet-tdt-0.6b-v2.nemo"
    if not nemo_path.is_file():
        print(f"找不到 NeMo 模型文件: {nemo_path}")
        sys.exit(1)

    print(f"[NeMo] restore_from: {nemo_path}")
    nemo_model = EncDecRNNTBPEModel.restore_from(str(nemo_path), map_location=device)
    nemo_model = nemo_model.to(device)
    nemo_model.eval()
    return nemo_model


def _load_local_parakeet(device: torch.device):
    from nemoasr2pytorch.asr.api import load_default_parakeet_tdt_model

    model = load_default_parakeet_tdt_model(device=device, dtype=torch.float32)
    model.eval()
    return model


def _load_waveform(wav_path: Path, device: torch.device, target_sr: int = 16000) -> tuple[torch.Tensor, torch.Tensor]:
    import torchaudio

    if not wav_path.is_file():
        print(f"找不到音频文件: {wav_path}")
        sys.exit(1)

    sig, sr = torchaudio.load(str(wav_path))
    if sig.size(0) > 1:
        sig = sig.mean(dim=0, keepdim=True)
    sig = sig.squeeze(0)
    if sr != target_sr:
        sig = torchaudio.functional.resample(sig, orig_freq=sr, new_freq=target_sr)
    sig = sig.to(device)
    length = torch.tensor([sig.numel()], device=device, dtype=torch.long)
    return sig.unsqueeze(0), length  # [1,T], [1]


def main() -> None:
    wav_path = REPO_ROOT / "test_01.wav"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    nemo_model = _load_nemo_parakeet(device)
    local_model = _load_local_parakeet(device)

    # 加载波形
    wav, length = _load_waveform(wav_path, device=device, target_sr=16000)
    print(f"waveform shape: {wav.shape}, length(samples)={int(length[0])}")

    with torch.no_grad():
        # NeMo 路径
        nemo_feats, nemo_feat_len = nemo_model.preprocessor(input_signal=wav, length=length)
        # subsampling 之后的特征（对照 ConvSubsampling）
        nemo_sub_in = nemo_feats.transpose(1, 2)  # [B,T,F]
        nemo_sub, nemo_sub_len = nemo_model.encoder.pre_encode(x=nemo_sub_in, lengths=nemo_feat_len)
        nemo_enc, nemo_enc_len = nemo_model.encoder(audio_signal=nemo_feats, length=nemo_feat_len)

        # 本地实现路径
        local_feats, local_feat_len = local_model.preprocessor(wav, length)
        local_sub_in = local_feats.transpose(1, 2)
        local_sub, local_sub_len = local_model.encoder.pre_encode(local_sub_in, local_feat_len)
        local_enc, local_enc_len = local_model.encoder(local_feats, local_feat_len)

    # 对齐长度
    min_feat_T = int(min(nemo_feats.size(-1), local_feats.size(-1)))
    min_enc_T = int(min(nemo_enc.size(-1), local_enc.size(-1)))

    # subsampling 对齐
    min_sub_T = int(min(nemo_sub.size(1), local_sub.size(1)))
    nemo_sub_c = nemo_sub[:, :min_sub_T, :]
    local_sub_c = local_sub[:, :min_sub_T, :]

    nemo_feats_c = nemo_feats[..., :min_feat_T]
    local_feats_c = local_feats[..., :min_feat_T]
    nemo_enc_c = nemo_enc[..., :min_enc_T]
    local_enc_c = local_enc[..., :min_enc_T]

    # 统计信息
    print("\n=== Mel 特征对比 ===")
    print(f"NeMo feats shape   : {tuple(nemo_feats.shape)}, len={int(nemo_feat_len[0])}")
    print(f"Local feats shape  : {tuple(local_feats.shape)}, len={int(local_feat_len[0])}")
    feat_l1 = (nemo_feats_c - local_feats_c).abs().mean().item()
    feat_l2 = torch.sqrt(((nemo_feats_c - local_feats_c) ** 2).mean()).item()
    print(f"Mel L1 mean diff   : {feat_l1:.6f}")
    print(f"Mel L2 mean diff   : {feat_l2:.6f}")

    print("\n=== Subsampling 输出对比（encoder 之前） ===")
    print(f"NeMo sub shape     : {tuple(nemo_sub.shape)}, len={int(nemo_sub_len[0])}")
    print(f"Local sub shape    : {tuple(local_sub.shape)}, len={int(local_sub_len[0])}")
    sub_l1 = (nemo_sub_c - local_sub_c).abs().mean().item()
    sub_l2 = torch.sqrt(((nemo_sub_c - local_sub_c) ** 2).mean()).item()
    print(f"Sub L1 mean diff   : {sub_l1:.6f}")
    print(f"Sub L2 mean diff   : {sub_l2:.6f}")

    print("\n=== Encoder 输出对比 ===")
    print(f"NeMo enc shape     : {tuple(nemo_enc.shape)}, len={int(nemo_enc_len[0])}")
    print(f"Local enc shape    : {tuple(local_enc.shape)}, len={int(local_enc_len[0])}")
    enc_l1 = (nemo_enc_c - local_enc_c).abs().mean().item()
    enc_l2 = torch.sqrt(((nemo_enc_c - local_enc_c) ** 2).mean()).item()
    print(f"Enc L1 mean diff   : {enc_l1:.6f}")
    print(f"Enc L2 mean diff   : {enc_l2:.6f}")

    # 打印若干时间步的 encoder 向量差异（方便发现是否后半段偏得更厉害）
    print("\n=== 部分时间步的 Encoder 向量范数 ===")
    for frac in [0.25, 0.5, 0.75]:
        t = int(min_enc_T * frac)
        nemo_vec = nemo_enc_c[0, :, t]
        local_vec = local_enc_c[0, :, t]
        print(f"t={t:4d}: |NeMo|={nemo_vec.norm().item():.4f}, |Local|={local_vec.norm().item():.4f}, |diff|={(nemo_vec-local_vec).norm().item():.4f}")


if __name__ == "__main__":
    main()
