# ACE-Step 1.5 安装指南

**Language / 语言 / 言語:** [English](../en/INSTALL.md) | [中文](INSTALL.md) | [日本語](../ja/INSTALL.md)

---

## 目录

- [环境要求](#环境要求)
- [快速开始（全平台）](#快速开始全平台)
- [Windows 便携包](#-windows-便携包)
- [AMD / ROCm 显卡](#amd--rocm-显卡)
- [Intel 显卡](#intel-显卡)
- [仅 CPU 模式](#仅-cpu-模式)
- [Linux 注意事项](#linux-注意事项)
- [环境变量 (.env)](#环境变量-env)
- [命令行参数](#命令行参数)
- [模型下载](#-模型下载)
- [如何选择模型？](#-如何选择模型)
- [开发](#开发)

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.11+（正式版，非预发布版） |
| GPU | 推荐 CUDA GPU；也支持 MPS / ROCm / Intel XPU / CPU |
| 显存 | 仅 DiT 模式 ≥4GB；LLM+DiT ≥6GB |
| 磁盘 | 核心模型约 10GB |

---

## 快速开始（全平台）

### 1. 安装 uv（包管理器）

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 克隆 & 安装

```bash
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5
uv sync
```

### 3. 启动

**Gradio 网页界面（推荐）：**

```bash
uv run acestep
```

**REST API 服务器：**

```bash
uv run acestep-api
```

**直接使用 Python**（Conda / venv / 系统 Python）：

```bash
# 先激活你的环境，然后：
python acestep/acestep_v15_pipeline.py          # Gradio UI
python acestep/api_server.py                     # REST API
```

> 首次运行时模型会自动下载。打开 http://localhost:7860（Gradio）或 http://localhost:8001（API）。

---

## 🪟 Windows 便携包

为 Windows 用户提供了预装依赖的便携包：

1. 下载并解压：[ACE-Step-1.5.7z](https://files.acemusic.ai/acemusic/win/ACE-Step-1.5.7z)
2. 包含 `python_embeded`，所有依赖已预装
3. **要求：** CUDA 12.8

### 快速启动脚本

| 脚本 | 说明 |
|------|------|
| `start_gradio_ui.bat` | 启动 Gradio 网页界面 |
| `start_api_server.bat` | 启动 REST API 服务器 |

两个脚本均支持自动环境检测、自动安装 `uv`、可配置下载源、可选 Git 更新检查、可自定义模型和参数。

### 配置

**`start_gradio_ui.bat`：**

```batch
REM 界面语言 (en, zh, he, ja)
set LANGUAGE=zh

REM 下载源 (auto, huggingface, modelscope)
set DOWNLOAD_SOURCE=--download-source modelscope

REM Git 更新检查 (true/false)
set CHECK_UPDATE=true

REM 模型配置
set CONFIG_PATH=--config_path acestep-v15-turbo
set LM_MODEL_PATH=--lm_model_path acestep-5Hz-lm-1.7B
```

### 更新与维护

| 脚本 | 用途 |
|------|------|
| `check_update.bat` | 从 GitHub 检查并更新 |
| `merge_config.bat` | 更新后合并备份的配置 |
| `install_uv.bat` | 安装 uv 包管理器 |
| `quick_test.bat` | 测试环境配置 |

---

## AMD / ROCm 显卡

> ⚠️ `uv run acestep` 会安装 CUDA PyTorch wheels，可能覆盖已有的 ROCm 环境。

### 推荐工作流

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate

# 2. 安装 ROCm 兼容的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# 3. 安装 ACE-Step
pip install -e .

# 4. 启动服务
python -m acestep.acestep_v15_pipeline --port 7680
```

### GPU 检测问题排查

如果显示 "No GPU detected, running on CPU"：

1. 运行诊断工具：`python scripts/check_gpu.py`
2. RDNA3 GPU 设置 `HSA_OVERRIDE_GFX_VERSION`：

| GPU | 值 |
|-----|---|
| RX 7900 XT/XTX, RX 9070 XT | `export HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| RX 7800 XT, RX 7700 XT | `export HSA_OVERRIDE_GFX_VERSION=11.0.1` |
| RX 7600 | `export HSA_OVERRIDE_GFX_VERSION=11.0.2` |

3. Windows 上使用 `start_gradio_ui_rocm.bat`
4. 验证 ROCm 安装：`rocm-smi`

### Linux（cachy-os / RDNA4）

详见 [ACE-Step1.5-Rocm-Manual-Linux.md](../en/ACE-Step1.5-Rocm-Manual-Linux.md)

---

## Intel 显卡

| 项目 | 详情 |
|------|------|
| 测试设备 | Windows 笔记本，Ultra 9 285H 集成显卡 |
| 卸载 | 默认禁用 |
| 编译与量化 | 默认启用 |
| LLM 推理 | 支持（已测试 `acestep-5Hz-lm-0.6B`） |
| nanovllm 加速 | Intel GPU 暂不支持 |
| 测试环境 | PyTorch 2.8.0（[Intel Extension for PyTorch](https://pytorch-extension.intel.com/?request=platform)） |

> 注意：生成超过 2 分钟的音频时，LLM 推理速度可能下降。Intel 独立显卡预计可用但尚未测试。

---

## 仅 CPU 模式

ACE-Step 可以在 CPU 上运行**仅推理**，但速度会显著变慢。

- 不推荐在 CPU 上训练（包括 LoRA）。
- 低显存系统可使用 DiT-only 模式（禁用 LLM）。

如果没有 GPU，建议：
- 使用云 GPU 服务
- 仅运行推理工作流
- 使用 `ACESTEP_INIT_LLM=false` 启用 DiT-only 模式

---

## Linux 注意事项

### Python 3.11 预发布版问题

部分 Linux 发行版（包括 Ubuntu）自带 Python 3.11.0rc1 预发布版，可能导致 vLLM 后端出现段错误。

**建议：** 使用稳定版 Python（≥ 3.11.12）。Ubuntu 上可通过 deadsnakes PPA 安装。

如无法升级 Python，使用 PyTorch 后端：

```bash
uv run acestep --backend pt
```

---

## 环境变量 (.env)

```bash
cp .env.example .env   # 复制并编辑
```

### 关键变量

| 变量 | 取值 | 说明 |
|------|------|------|
| `ACESTEP_INIT_LLM` | `auto` / `true` / `false` | LLM 初始化模式 |
| `ACESTEP_CONFIG_PATH` | 模型名称 | DiT 模型路径 |
| `ACESTEP_LM_MODEL_PATH` | 模型名称 | LM 模型路径 |
| `ACESTEP_DOWNLOAD_SOURCE` | `auto` / `huggingface` / `modelscope` | 下载源 |
| `ACESTEP_API_KEY` | 字符串 | API 认证密钥 |

### LLM 初始化 (`ACESTEP_INIT_LLM`)

处理流程：`GPU 检测 → ACESTEP_INIT_LLM 覆盖 → 模型加载`

| 值 | 行为 |
|----|------|
| `auto`（或空） | 使用 GPU 自动检测结果（推荐） |
| `true` / `1` / `yes` | 强制启用 LLM（可能导致 OOM） |
| `false` / `0` / `no` | 强制禁用，纯 DiT 模式 |

**示例 `.env`：**

```bash
# 自动模式（推荐）
ACESTEP_INIT_LLM=auto

# 低显存 GPU 强制启用
ACESTEP_INIT_LLM=true
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B

# 禁用 LLM 加速生成
ACESTEP_INIT_LLM=false
```

---

## 命令行参数

### Gradio UI (`acestep`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 7860 | 服务端口 |
| `--server-name` | 127.0.0.1 | 服务地址（使用 `0.0.0.0` 开放网络访问） |
| `--share` | false | 创建公开 Gradio 链接 |
| `--language` | en | 界面语言：`en`、`zh`、`he`、`ja` |
| `--init_service` | false | 启动时自动初始化模型 |
| `--init_llm` | auto | LLM 初始化：`true` / `false` / 省略为自动 |
| `--config_path` | auto | DiT 模型（如 `acestep-v15-turbo`） |
| `--lm_model_path` | auto | LM 模型（如 `acestep-5Hz-lm-1.7B`） |
| `--offload_to_cpu` | auto | CPU 卸载（显存 < 16GB 时自动启用） |
| `--download-source` | auto | 模型源：`auto` / `huggingface` / `modelscope` |
| `--enable-api` | false | 同时启用 REST API 端点 |

**示例：**

```bash
# 公开访问 + 中文界面
uv run acestep --server-name 0.0.0.0 --share --language zh

# 启动时预初始化模型
uv run acestep --init_service true --config_path acestep-v15-turbo

# 使用 ModelScope 下载
uv run acestep --download-source modelscope
```

---

## 📥 模型下载

首次运行时模型会从 [HuggingFace](https://huggingface.co/ACE-Step/Ace-Step1.5) 或 [ModelScope](https://modelscope.cn/organization/ACE-Step) 自动下载。

### CLI 下载

```bash
uv run acestep-download                              # 下载主模型
uv run acestep-download --all                         # 下载所有模型
uv run acestep-download --download-source modelscope  # 从 ModelScope 下载
uv run acestep-download --model acestep-v15-sft       # 指定模型
uv run acestep-download --list                        # 列出所有可用模型
```

### 手动下载 (huggingface-cli)

```bash
# 主模型
huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir ./checkpoints

# 可选模型
huggingface-cli download ACE-Step/acestep-5Hz-lm-0.6B --local-dir ./checkpoints/acestep-5Hz-lm-0.6B
huggingface-cli download ACE-Step/acestep-5Hz-lm-4B --local-dir ./checkpoints/acestep-5Hz-lm-4B
```

### 可用模型

| 模型 | 说明 | HuggingFace |
|------|------|-------------|
| **Ace-Step1.5**（主模型） | 核心：vae, Qwen3-Embedding-0.6B, acestep-v15-turbo, acestep-5Hz-lm-1.7B | [链接](https://huggingface.co/ACE-Step/Ace-Step1.5) |
| acestep-5Hz-lm-0.6B | 轻量 LM（0.6B 参数） | [链接](https://huggingface.co/ACE-Step/acestep-5Hz-lm-0.6B) |
| acestep-5Hz-lm-4B | 大型 LM（4B 参数） | [链接](https://huggingface.co/ACE-Step/acestep-5Hz-lm-4B) |
| acestep-v15-base | 基础 DiT 模型 | [链接](https://huggingface.co/ACE-Step/acestep-v15-base) |
| acestep-v15-sft | SFT DiT 模型 | [链接](https://huggingface.co/ACE-Step/acestep-v15-sft) |
| acestep-v15-turbo-shift1 | Turbo DiT（shift1） | [链接](https://huggingface.co/ACE-Step/acestep-v15-turbo-shift1) |
| acestep-v15-turbo-shift3 | Turbo DiT（shift3） | [链接](https://huggingface.co/ACE-Step/acestep-v15-turbo-shift3) |
| acestep-v15-turbo-continuous | Turbo DiT（continuous shift 1-5） | [链接](https://huggingface.co/ACE-Step/acestep-v15-turbo-continuous) |

---

## 💡 如何选择模型？

ACE-Step 会自动适配你的 GPU 显存：

| GPU 显存 | 推荐 LM 模型 | 说明 |
|----------|--------------|------|
| **≤6GB** | 无（仅 DiT） | 默认禁用 LM 以节省显存 |
| **6-12GB** | `acestep-5Hz-lm-0.6B` | 轻量，平衡性好 |
| **12-16GB** | `acestep-5Hz-lm-1.7B` | 更好的质量 |
| **≥16GB** | `acestep-5Hz-lm-4B` | 最佳质量和音频理解能力 |

> 📖 详细 GPU 兼容性信息（时长限制、批量大小、内存优化），请参阅 [GPU 兼容性指南](GPU_COMPATIBILITY.md)。

---

## 开发

```bash
# 添加依赖
uv add package-name
uv add --dev package-name

# 更新所有依赖
uv sync --upgrade
```
