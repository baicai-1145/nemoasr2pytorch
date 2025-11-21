# nemoasr2pytorch

Pure‑PyTorch inference port of several NeMo ASR models, with a focus on **Windows / WSL** support and **no NeMo runtime dependency**.

Currently supported:

- Frame‑level VAD: `Frame_VAD_Multilingual_MarbleNet_v2.0`
- ASR (RNNT‑TDT):
  - `parakeet-tdt-0.6b-v2` – English
  - `parakeet-tdt-0.6b-v3` – Multilingual

The project only targets **inference** – no training or data pipelines – and mirrors NeMo’s architecture closely so that results match NeMo as much as possible.

## Installation

1. Install a suitable **PyTorch + torchaudio** build first (GPU or CPU), following the official instructions.  
   For example, on CUDA 12.6:

   ```bash
   pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
     --index-url https://download.pytorch.org/whl/cu126
   ```

2. Then install this package:

   ```bash
   pip install nemoasr2pytorch
   ```

> Torch is **not** pinned as a dependency on purpose – you stay in control of the exact CUDA / CPU build.

## Quick ASR usage (Parakeet‑TDT)

The simplest way to run ASR on a single WAV file:

```python
from nemoasr2pytorch.asr.api import load_default_parakeet_tdt_model, transcribe

# lang="EN" -> parakeet-tdt-0.6b-v2 (English)
# lang="EU" -> parakeet-tdt-0.6b-v3 (multilingual)
model = load_default_parakeet_tdt_model(lang="EU")

text = transcribe(model, "your_audio.wav")
print(text)
```

Details:

- On first use, the corresponding `.pt` weights are automatically downloaded from ModelScope
  and cached under `exports/parakeet_tdt_0.6b_v{2,3}.pt` in your working directory.
- Subsequent runs reuse the local `.pt` directly.

### Low‑precision inference (FP16 / BF16)

On GPU you can load the model directly in low precision to save memory:

```python
from nemoasr2pytorch.asr.api import (
    load_parakeet_tdt_fp16,
    load_parakeet_tdt_bf16,
    transcribe_amp,
)

# FP16 model (GPU only)
model_fp16 = load_parakeet_tdt_fp16(lang="EU")
print("FP16:", transcribe_amp(model_fp16, "your_audio.wav"))

# BF16 model (if hardware supports it)
model_bf16 = load_parakeet_tdt_bf16(lang="EU")
print("BF16:", transcribe_amp(model_bf16, "your_audio.wav"))
```

`transcribe_amp` uses PyTorch AMP (`torch.amp.autocast`) on CUDA to run the model in mixed precision.

## VAD (MarbleNet) for pre‑segmentation

Frame‑level VAD API:

```python
from nemoasr2pytorch.vad.api import load_default_frame_vad_model, run_vad_on_waveform
import torchaudio

# Loads MarbleNet VAD; if the .pt is missing, it is auto-downloaded
# from ModelScope to ./exports/frame_vad_multilingual_marblenet_v2.0.pt
vad_model = load_default_frame_vad_model()

waveform, sr = torchaudio.load("your_audio.wav")
if sr != vad_model.preprocessor.sample_rate:
    waveform = torchaudio.functional.resample(
        waveform, sr, vad_model.preprocessor.sample_rate
    )

probs, segments = run_vad_on_waveform(vad_model, waveform.squeeze(0))
print("Segments:", segments)
```

## Long‑audio inference (concept)

The repository version ships a reference script `inference.py` which:

- loads a Parakeet model (v2/v3, chosen by `lang`);
- optionally runs MarbleNet VAD to detect speech regions;
- merges VAD segments into chunks based on `min_seg` / `max_seg` length;
- runs Parakeet on each chunk and concatenates the results.

The core logic is implemented via the public APIs:

- `nemoasr2pytorch.vad.api` – VAD model + `run_vad_on_waveform`
- `nemoasr2pytorch.asr.api` – Parakeet model + `transcribe` / `transcribe_amp`

You can either:

- copy `inference.py` from the GitHub repo and adapt it to your own CLI; or
- re‑implement a similar pipeline in your application using the two APIs above.

## Package APIs

Main public modules:

- `nemoasr2pytorch.asr.api`
  - `load_default_parakeet_tdt_model(lang="EN" | "EU", device=None, dtype=torch.float32)`  
    Load Parakeet‑TDT in FP32; `lang` chooses v2 (EN) vs v3 (EU).  
  - `load_parakeet_tdt_fp16(lang="EN" | "EU", device=None)`  
    Load FP16 model (usually on GPU).  
  - `load_parakeet_tdt_bf16(lang="EN" | "EU", device=None)`  
    Load BF16 model (if supported).  
  - `transcribe(model, audio)`  
    Greedy TDT decoding in full precision (CPU or GPU).  
  - `transcribe_amp(model, audio)`  
    Greedy TDT decoding with AMP on CUDA for low‑precision models.

- `nemoasr2pytorch.vad.api`
  - `load_default_frame_vad_model(device=None, dtype=torch.float32)`  
    Load the MarbleNet VAD model from a local `.pt`.  
  - `run_vad_on_waveform(model, audio, ...)`  
    Compute per‑frame speech probabilities and return merged speech segments.

## Notes / Limitations

- This package focuses on **inference only**; training and NeMo’s full config stack (Hydra/Lightning) are intentionally omitted.
- Parakeet weights (`.pt`) are auto‑downloaded from ModelScope on first use; VAD `.pt` is currently expected to be provided by the user (converted from NeMo).
- For best performance and lower memory usage, a CUDA‑enabled PyTorch build is recommended; CPU‑only inference also works but will be slower on long audio.
