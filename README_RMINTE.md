# RMinte-Orin-TensorRT-EDGE-LLM

基于 [NVIDIA TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) 的 **Jetson AGX Orin** 高性能 VLM 推理方案，专为边缘设备上的视觉语言模型部署而优化。

## 🚀 主要特性

- **OpenAI 兼容 API 服务器** - 完整支持 `/v1/chat/completions` 端点
- **真正的流式输出** - Token 级别的 SSE 流式响应，支持 UTF-8 多字节字符
- **Base64 图片支持** - 直接处理前端传来的 Base64 编码图片
- **自动图片缩放** - 自动调整大图以适应 token 限制
- **持久化引擎** - 引擎一次加载，常驻内存，避免重复加载
- **INT4-AWQ 量化支持** - 支持量化模型以减少显存占用

## 📦 新增组件

### 1. OpenAI 兼容 HTTP 服务器

**文件位置**: `examples/server/llm_server.cpp`

功能特性:
- 完整的 OpenAI Chat Completions API 兼容
- 支持流式 (SSE) 和非流式响应
- 支持多模态输入 (文本 + 图片)
- Base64 图片数据 URL 自动解码
- 大图自动缩放 (默认最大 896px)
- 持久化引擎，启动后常驻内存

### 2. 流式推理 API

**文件位置**: `cpp/runtime/llmInferenceRuntime.cpp`

新增 `handleRequestStreaming()` 方法:
- Token 级别的回调机制
- UTF-8 字符边界正确处理
- 支持停止词检测

## 🛠️ 构建指南

### 环境要求

- **硬件**: Jetson AGX Orin (64GB 推荐)
- **系统**: JetPack 6.2+ (L4T R36.4.x)
- **TensorRT**: 10.7+
- **CUDA**: 12.6

### 编译步骤

```bash
# 克隆仓库
git clone https://github.com/thomas-hiddenpeak/RMinte-Orin-TensorRT-EDGE-LLM.git
cd RMinte-Orin-TensorRT-EDGE-LLM

# 创建构建目录
mkdir -p build && cd build

# 配置 CMake (Jetson AGX Orin)
cmake .. -DTRT_PACKAGE_DIR=/usr -DCUDA_VERSION=12.6 -DCMAKE_CUDA_ARCHITECTURES=87

# 编译
make -j$(nproc)
```

### 编译产物

- `build/examples/llm/llm_build` - LLM 引擎构建工具
- `build/examples/multimodal/visual_build` - 视觉编码器引擎构建工具
- `build/examples/server/llm_server` - OpenAI 兼容 API 服务器

## 📖 使用指南

### 1. 导出 ONNX 模型

```bash
# 安装 Python 包
pip install -e .

# 导出 LLM
tensorrt-edgellm-export-llm \
  --model=/path/to/Qwen3-VL-8B-Instruct \
  --output=/path/to/onnx/llm/Qwen3-VL-8B-Instruct

# 导出视觉编码器
tensorrt-edgellm-export-visual \
  --model=/path/to/Qwen3-VL-8B-Instruct \
  --output=/path/to/onnx/visual/Qwen3-VL-8B-Instruct
```

### 2. 构建 TensorRT 引擎

```bash
cd build
export EDGELLM_PLUGIN_PATH=$PWD/libNvInfer_edgellm_plugin.so

# 构建 LLM 引擎
./examples/llm/llm_build \
  --onnxDir=/path/to/onnx/llm/Qwen3-VL-8B-Instruct \
  --engineDir=/path/to/engine/llm/Qwen3-VL-8B-Instruct \
  --maxInputLen=16384 \
  --maxKVCacheCapacity=32768 \
  --maxBatchSize=4 \
  --vlm \
  --minImageTokens=256 \
  --maxImageTokens=4096

# 构建视觉编码器引擎
./examples/multimodal/visual_build \
  --onnxDir=/path/to/onnx/visual/Qwen3-VL-8B-Instruct \
  --engineDir=/path/to/engine/visual/Qwen3-VL-8B-Instruct
```

### 3. 启动 API 服务器

```bash
cd build
export EDGELLM_PLUGIN_PATH=$PWD/libNvInfer_edgellm_plugin.so

./examples/server/llm_server \
  --engineDir=/path/to/engine/llm/Qwen3-VL-8B-Instruct \
  --multimodalEngineDir=/path/to/engine/visual/Qwen3-VL-8B-Instruct \
  --port=58010 \
  --debug
```

### 4. API 使用示例

#### 文本对话

```bash
curl -X POST http://localhost:58010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "stream": true
  }'
```

#### 图片理解 (URL)

```bash
curl -X POST http://localhost:58010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
          {"type": "text", "text": "描述这张图片"}
        ]
      }
    ],
    "stream": true
  }'
```

#### 图片理解 (Base64)

```bash
curl -X POST http://localhost:58010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}},
          {"type": "text", "text": "这是什么?"}
        ]
      }
    ],
    "stream": true
  }'
```

## ⚙️ 配置参数说明

### 引擎构建参数

| 参数 | 说明 | 推荐值 (64GB Orin) |
|------|------|-------------------|
| `--maxInputLen` | 最大输入长度 (tokens) | 16384 |
| `--maxKVCacheCapacity` | KV 缓存容量 (tokens) | 32768 |
| `--maxBatchSize` | 最大批处理大小 | 4 |
| `--maxImageTokens` | 单图最大 tokens | 4096 |
| `--minImageTokens` | 单图最小 tokens | 256 |

### 服务器参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--engineDir` | LLM 引擎目录 | (必需) |
| `--multimodalEngineDir` | 视觉引擎目录 | (可选) |
| `--port` | 服务端口 | 58010 |
| `--debug` | 调试模式 | false |

## 📊 性能参考

在 Jetson AGX Orin 64GB 上测试 Qwen3-VL-8B-Instruct:

| 配置 | 引擎大小 | 加载时间 | Token 速度 |
|------|---------|---------|-----------|
| FP16 | ~16 GB | ~28s | ~30 tok/s |
| INT4-AWQ | ~6 GB | ~14s | ~45 tok/s |

## 🔧 修改的文件

相比原始 TensorRT-Edge-LLM，本项目修改/新增了以下文件:

### 新增文件
- `examples/server/llm_server.cpp` - OpenAI 兼容 API 服务器
- `examples/server/CMakeLists.txt` - 服务器构建配置
- `.github/copilot-instructions.md` - 项目说明

### 修改文件
- `cpp/runtime/llmInferenceRuntime.cpp` - 添加 `handleRequestStreaming()` 流式推理
- `cpp/runtime/llmInferenceRuntime.h` - 添加流式 API 声明
- `cpp/common/tensor.cpp` - Tensor 工具修复
- `examples/CMakeLists.txt` - 添加 server 子目录

## 🤝 与前端集成

本服务器兼容以下客户端:

- **ChatBox** - 推荐，完美支持
- **OpenWebUI** - 支持
- **其他 OpenAI 兼容客户端**

配置示例 (ChatBox):
- API 地址: `http://<orin-ip>:58010`
- API 路径: `/v1/chat/completions`
- 模型名称: `qwen3-vl`

## 📝 许可证

本项目基于 Apache-2.0 许可证，继承自 NVIDIA TensorRT-Edge-LLM。

## 🙏 致谢

- [NVIDIA TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) - 原始项目
- [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) - 视觉语言模型
