# nemoasr2pytorch

一个将部分 NeMo ASR 推理能力移植到纯 PyTorch 的项目，目标是在 **不依赖 NeMo 作为运行时库** 的情况下，在 Windows / WSL / Linux 上直接完成推理：

- Frame_VAD_Multilingual_MarbleNet_v2.0 帧级 VAD
- parakeet-tdt-0.6b-v2 RNNT‑TDT 语音识别
- parakeet-tdt-0.6b-v3 RNNT‑TDT 语音识别

本项目只关注 **推理**，不实现训练、数据集管道或复杂配置系统，尽量复用 NeMo 源码逻辑保证推理结果对齐。

## 安装

1. 先按照官方文档安装 **合适的 PyTorch / torchaudio 版本**，例如 CUDA 12.6 环境下：

   ```bash
   pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
   ```

   如需 CPU 版本或其他 CUDA 版本，请参考 https://pytorch.org/get-started/previous-versions/。

2. 再安装本项目其余依赖：

   ```bash
   pip install -r requirements.txt
   ```

## 快速上手

```bash
(.venv) python inference.py --wav test.wav --lang EU --precision fp16
```

- `inference.py`：正式推理脚本，支持两种模式：
  - 先使用 MarbleNet VAD 进行长音频切分，再按语言/精度预设调用 parakeet-tdt 推理；
  - 或使用 `--no-vad` 直接按固定长度切分整段音频进行 ASR。

## 代码入口概览

核心 Python 包为 `nemoasr2pytorch`，主要入口如下：

- `nemoasr2pytorch/vad/api.py`  
  - `load_default_frame_vad_model()`：加载 MarbleNet VAD 模型（若本地缺失会自动从 ModelScope 下载到 `./exports`）。  
  - `frame_vad(...)`：对输入波形做帧级 VAD 推理。  
- `nemoasr2pytorch/asr/api.py`  
  - `load_default_parakeet_tdt_model(lang=\"EN\" | \"EU\")`：以 FP32 加载 parakeet-tdt 模型，按语言预设选择 v2（EN）或 v3（EU）。  
  - `load_parakeet_tdt_fp16(lang=...)` / `load_parakeet_tdt_bf16(lang=...)`：在 GPU 上以低精度加载对应语言预设的模型。  
  - `transcribe(...)` / `transcribe_amp(...)`：完成一次音频转写（带/不带自动混合精度）。  

根目录的 `inference.py` 则作为高层封装，将 VAD + ASR 串联起来用于长音频推理，并提供：

- `--lang EN|EU`：切换英文 / 多语言 parakeet 预设；  
- `--precision fp32|fp16|bf16`：选择推理精度（GPU 上 FP16/BF16 可降低显存占用）；  
- `--no-vad` / `--min-seg` / `--max-seg`：控制是否使用 VAD 及切分策略；  
- `--cpu-only`：强制在 CPU 上推理，便于在无 GPU 环境测试。  

更细节的模型结构（Conformer encoder、RNNT decoder、TDT greedy 解码等）均位于 `nemoasr2pytorch/models` 与 `nemoasr2pytorch/decoding`，基本是一一对应 NeMo 源码的纯 PyTorch 端口，方便后续进一步扩展或调试。
