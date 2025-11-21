from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from nemoasr2pytorch.models.asr.parakeet_tdt import ParakeetTDTModel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAKEET_V2_PT = REPO_ROOT / "exports/parakeet_tdt_0.6b_v2.pt"
DEFAULT_PARAKEET_V3_PT = REPO_ROOT / "exports/parakeet_tdt_0.6b_v3.pt"


def _get_parakeet_pt_by_lang(lang: str) -> Path:
    """
    根据语言预设选择对应的 parakeet-tdt .pt 文件。

    - 'EN'：parakeet-tdt-0.6b-v2（英文）
    - 'EU'：parakeet-tdt-0.6b-v3（多语言 / 欧洲语言）
    """
    key = lang.upper()
    if key == "EN":
        return DEFAULT_PARAKEET_V2_PT
    if key == "EU":
        return DEFAULT_PARAKEET_V3_PT
    raise ValueError(f"Unsupported language preset: {lang!r}. Expected 'EN' or 'EU'.")


def _get_parakeet_remote_urls(lang: str) -> list[str]:
    """
    返回对应语言预设在 ModelScope 上的下载 URL 列表（按优先级顺序）。

    - v2（EN）：尝试 resolve/master 与 file/view/master 两种形式，增加兼容性。
    - v3（EU）：使用用户提供的 resolve/master URL。
    """
    base = "https://www.modelscope.cn/models/baicai1145/nemoasr2pytorch"
    key = lang.upper()
    if key == "EN":
        return [
            f"{base}/resolve/master/parakeet_tdt_0.6b_v2.pt",
            f"{base}/file/view/master/parakeet_tdt_0.6b_v2.pt",
        ]
    if key == "EU":
        return [
            f"{base}/resolve/master/parakeet_tdt_0.6b_v3.pt",
        ]
    raise ValueError(f"Unsupported language preset for remote URL: {lang!r}. Expected 'EN' or 'EU'.")


def _ensure_parakeet_pt(lang: str) -> Path:
    """
    确保本地存在对应语言预设的 parakeet .pt 文件：
    - 若本地已存在则直接返回；
    - 若不存在则尝试从 ModelScope 自动下载。
    """
    pt_path = _get_parakeet_pt_by_lang(lang)
    if pt_path.is_file():
        return pt_path

    urls = _get_parakeet_remote_urls(lang)
    pt_path.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for url in urls:
        try:
            import urllib.request

            # 简单下载，依赖用户环境的网络配置
            print(f"[nemoasr2pytorch] Downloading Parakeet ({lang}) model from {url} ...")
            urllib.request.urlretrieve(url, pt_path)
            print(f"[nemoasr2pytorch] Saved model to {pt_path}")
            return pt_path
        except Exception as e:  # pragma: no cover - 网络环境依赖
            last_err = e
            continue

    if last_err is not None:
        raise FileNotFoundError(
            f"Failed to download Parakeet .pt model for lang={lang!r} to {pt_path}.\n"
            f"Tried URLs: {urls}\n"
            f"Last error: {last_err}\n"
            "Please check your network / proxy settings, or manually download the .pt file "
            "from ModelScope and place it at the expected path."
        ) from last_err

    return pt_path


def load_default_parakeet_tdt_model(
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lang: str = "EN",
) -> ParakeetTDTModel:
    """
    加载 parakeet-tdt 模型（仅支持 .pt），支持语言预设：

    - lang='EN'：parakeet-tdt-0.6b-v2 英文模型（exports/parakeet_tdt_0.6b_v2.pt）
    - lang='EU'：parakeet-tdt-0.6b-v3 多语言模型（exports/parakeet_tdt_0.6b_v3.pt）

    使用前需要先用 examples/export_from_nemo_to_pt.py 从对应 .nemo 导出 .pt。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pt_path = _ensure_parakeet_pt(lang)
    model: ParakeetTDTModel = torch.load(pt_path, map_location=device, weights_only=False)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def load_parakeet_tdt_fp16(
    device: str | torch.device | None = None,
    lang: str = "EN",
) -> ParakeetTDTModel:
    """在 GPU 上以 FP16 加载 parakeet 模型，用于降低显存占用。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_default_parakeet_tdt_model(device=device, dtype=torch.float16, lang=lang)


def load_parakeet_tdt_bf16(
    device: str | torch.device | None = None,
    lang: str = "EN",
) -> ParakeetTDTModel:
    """以 BF16 加载 parakeet 模型（需硬件支持）。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_default_parakeet_tdt_model(device=device, dtype=torch.bfloat16, lang=lang)


def _to_waveform_tensor(x: Union[str, np.ndarray, torch.Tensor], target_sr: int) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if x.dim() == 2 and x.size(0) == 1:
            x = x.squeeze(0)
        if x.dim() != 1:
            raise ValueError("Only mono 1D waveform tensor is supported.")
        return x.float()

    if isinstance(x, np.ndarray):
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 1:
            raise ValueError("Only mono 1D numpy waveform is supported.")
        return torch.from_numpy(x.astype("float32"))

    import torchaudio

    wav_path = Path(x)
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    sig, sr = torchaudio.load(str(wav_path))
    if sig.size(0) > 1:
        sig = sig.mean(dim=0, keepdim=True)
    sig = sig.squeeze(0)
    if sr != target_sr:
        sig = torchaudio.functional.resample(sig, orig_freq=sr, new_freq=target_sr)
    return sig.float()


def transcribe(
    model: ParakeetTDTModel,
    audio: Union[str, np.ndarray, torch.Tensor],
) -> str:
    device = next(model.parameters()).device
    waveform = _to_waveform_tensor(audio, target_sr=model.sample_rate).to(device)
    return model.transcribe(waveform)


def transcribe_amp(
    model: ParakeetTDTModel,
    audio: Union[str, np.ndarray, torch.Tensor],
) -> str:
    """在 GPU 上使用 autocast 进行混合精度推理（适合 FP16/BF16 模型）。"""
    device = next(model.parameters()).device
    waveform = _to_waveform_tensor(audio, target_sr=model.sample_rate).to(device)

    if device.type == "cuda":
        # 根据模型权重 dtype 选择合适的 autocast dtype
        dtype = next(model.parameters()).dtype
        # 优先使用新的 torch.amp.autocast('cuda', ...) 接口，保持与 PyTorch 2.6+ 一致
        try:  # pragma: no cover - 分支取决于 PyTorch 版本
            autocast_ctx = torch.amp.autocast("cuda", dtype=dtype)  # type: ignore[arg-type]
        except Exception:
            autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
        with autocast_ctx:
            return model.transcribe(waveform)

    # CPU 上直接走普通路径
    return model.transcribe(waveform)
