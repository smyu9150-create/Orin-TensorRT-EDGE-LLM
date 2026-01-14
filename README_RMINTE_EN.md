# RMinte-Orin-TensorRT-EDGE-LLM

A high-performance VLM inference solution for **Jetson AGX Orin** based on [NVIDIA TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM), optimized for deploying Vision-Language Models on edge devices.

## 🚀 Key Features

- **OpenAI-Compatible API Server** - Full support for `/v1/chat/completions` endpoint
- **True Streaming Output** - Token-level SSE streaming responses with proper UTF-8 multi-byte character handling
- **Base64 Image Support** - Direct processing of Base64-encoded images from frontend clients
- **Auto Image Scaling** - Automatic resizing of large images to fit token limits
- **Persistent Engine** - Load engine once, keep in memory, avoid repeated loading
- **INT4-AWQ Quantization** - Support for quantized models to reduce memory footprint

## 📦 New Components

### 1. OpenAI-Compatible HTTP Server

**Location**: `examples/server/llm_server.cpp`

Features:
- Full OpenAI Chat Completions API compatibility
- Support for streaming (SSE) and non-streaming responses
- Multimodal input support (text + images)
- Automatic Base64 image data URL decoding
- Automatic large image resizing (default max 896px)
- Persistent engine, stays in memory after startup

### 2. Streaming Inference API

**Location**: `cpp/runtime/llmInferenceRuntime.cpp`

New `handleRequestStreaming()` method:
- Token-level callback mechanism
- Proper UTF-8 character boundary handling
- Stop word detection support

## 🛠️ Build Guide

### Requirements

- **Hardware**: Jetson AGX Orin (64GB recommended)
- **System**: JetPack 6.2+ (L4T R36.4.x)
- **TensorRT**: 10.7+
- **CUDA**: 12.6

### Build Steps

```bash
# Clone repository
git clone https://github.com/thomas-hiddenpeak/RMinte-Orin-TensorRT-EDGE-LLM.git
cd RMinte-Orin-TensorRT-EDGE-LLM

# Create build directory
mkdir -p build && cd build

# Configure CMake (Jetson AGX Orin)
cmake .. -DTRT_PACKAGE_DIR=/usr -DCUDA_VERSION=12.6 -DCMAKE_CUDA_ARCHITECTURES=87

# Build
make -j$(nproc)
```

### Build Artifacts

- `build/examples/llm/llm_build` - LLM engine builder
- `build/examples/multimodal/visual_build` - Visual encoder engine builder
- `build/examples/server/llm_server` - OpenAI-compatible API server

## 📖 Usage Guide

### 1. Export ONNX Models

```bash
# Install Python package
pip install -e .

# Export LLM
tensorrt-edgellm-export-llm \
  --model=/path/to/Qwen3-VL-8B-Instruct \
  --output=/path/to/onnx/llm/Qwen3-VL-8B-Instruct

# Export visual encoder
tensorrt-edgellm-export-visual \
  --model=/path/to/Qwen3-VL-8B-Instruct \
  --output=/path/to/onnx/visual/Qwen3-VL-8B-Instruct
```

### 2. Build TensorRT Engines

```bash
cd build
export EDGELLM_PLUGIN_PATH=$PWD/libNvInfer_edgellm_plugin.so

# Build LLM engine
./examples/llm/llm_build \
  --onnxDir=/path/to/onnx/llm/Qwen3-VL-8B-Instruct \
  --engineDir=/path/to/engine/llm/Qwen3-VL-8B-Instruct \
  --maxInputLen=16384 \
  --maxKVCacheCapacity=32768 \
  --maxBatchSize=4 \
  --vlm \
  --minImageTokens=256 \
  --maxImageTokens=4096

# Build visual encoder engine
./examples/multimodal/visual_build \
  --onnxDir=/path/to/onnx/visual/Qwen3-VL-8B-Instruct \
  --engineDir=/path/to/engine/visual/Qwen3-VL-8B-Instruct
```

### 3. Start API Server

```bash
cd build
export EDGELLM_PLUGIN_PATH=$PWD/libNvInfer_edgellm_plugin.so

./examples/server/llm_server \
  --engineDir=/path/to/engine/llm/Qwen3-VL-8B-Instruct \
  --multimodalEngineDir=/path/to/engine/visual/Qwen3-VL-8B-Instruct \
  --port=58010 \
  --debug
```

### 4. API Usage Examples

#### Text Conversation

```bash
curl -X POST http://localhost:58010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl",
    "messages": [
      {"role": "user", "content": "Hello, please introduce yourself"}
    ],
    "stream": true
  }'
```

#### Image Understanding (URL)

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
          {"type": "text", "text": "Describe this image"}
        ]
      }
    ],
    "stream": true
  }'
```

#### Image Understanding (Base64)

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
          {"type": "text", "text": "What is this?"}
        ]
      }
    ],
    "stream": true
  }'
```

## ⚙️ Configuration Parameters

### Engine Build Parameters

| Parameter | Description | Recommended (64GB Orin) |
|-----------|-------------|------------------------|
| `--maxInputLen` | Maximum input length (tokens) | 16384 |
| `--maxKVCacheCapacity` | KV cache capacity (tokens) | 32768 |
| `--maxBatchSize` | Maximum batch size | 4 |
| `--maxImageTokens` | Max tokens per image | 4096 |
| `--minImageTokens` | Min tokens per image | 256 |

### Server Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--engineDir` | LLM engine directory | (required) |
| `--multimodalEngineDir` | Visual engine directory | (optional) |
| `--port` | Server port | 58010 |
| `--debug` | Debug mode | false |

## 📊 Performance Reference

Tested on Jetson AGX Orin 64GB with Qwen3-VL-8B-Instruct:

| Configuration | Engine Size | Load Time | Token Speed |
|---------------|-------------|-----------|-------------|
| FP16 | ~16 GB | ~28s | ~30 tok/s |
| INT4-AWQ | ~6 GB | ~14s | ~45 tok/s |

## 🔧 Modified Files

Compared to the original TensorRT-Edge-LLM, this project modifies/adds the following files:

### New Files
- `examples/server/llm_server.cpp` - OpenAI-compatible API server
- `examples/server/CMakeLists.txt` - Server build configuration

### Modified Files
- `cpp/runtime/llmInferenceRuntime.cpp` - Added `handleRequestStreaming()` for streaming inference
- `cpp/runtime/llmInferenceRuntime.h` - Added streaming API declarations
- `cpp/common/tensor.cpp` - Tensor utility fixes
- `examples/CMakeLists.txt` - Added server subdirectory

## 🤝 Frontend Integration

This server is compatible with the following clients:

- **ChatBox** - Recommended, works perfectly
- **OpenWebUI** - Supported
- **Other OpenAI-compatible clients**

Configuration example (ChatBox):
- API URL: `http://<orin-ip>:58010`
- API Path: `/v1/chat/completions`
- Model Name: `qwen3-vl`

## 📝 License

This project is licensed under Apache-2.0, inherited from NVIDIA TensorRT-Edge-LLM.

## 🙏 Acknowledgments

- [NVIDIA TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) - Original project
- [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) - Vision-Language Model
