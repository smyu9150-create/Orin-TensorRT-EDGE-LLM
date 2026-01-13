# TensorRT Edge-LLM - Copilot Instructions

## Project Overview

TensorRT Edge-LLM is NVIDIA's high-performance C++ inference runtime for LLMs/VLMs on edge platforms (Jetson, DRIVE). Uses a **three-stage pipeline**: Python Export (HuggingFace→ONNX) → Engine Builder (ONNX→TensorRT) → C++ Runtime.

## Architecture

```
tensorrt_edgellm/     # Python: quantization, ONNX export (runs on host)
  ├── scripts/        # CLI entry points: tensorrt-edgellm-export-llm, etc.
  ├── quantization/   # FP8, INT4, NVFP4 quantization via nvidia-modelopt
  └── onnx_export/    # PyTorch → ONNX conversion
cpp/                  # C++ runtime (runs on edge device)
  ├── runtime/        # LlmInferenceRuntime, LlmInferenceSpecDecodeRuntime
  ├── builder/        # TensorRT engine compilation from ONNX
  ├── kernels/        # Custom CUDA kernels
  └── plugins/        # TensorRT plugins
examples/             # Reference implementations
  ├── llm/            # llm_build.cpp, llm_inference.cpp
  └── multimodal/     # VLM examples
```

## Build & Development

### C++ Build (requires TensorRT)
```bash
mkdir -p build && cd build
cmake .. -DTRT_PACKAGE_DIR=/path/to/tensorrt
make -j$(nproc)
```

### Native Build on Jetson AGX Orin (JetPack 6.2+)
```bash
# TensorRT is pre-installed at /usr on JetPack
mkdir -p build && cd build
cmake .. -DTRT_PACKAGE_DIR=/usr -DCUDA_VERSION=12.6 -DCMAKE_CUDA_ARCHITECTURES=87
make -j$(nproc)
# Outputs: build/examples/llm/llm_inference, build/examples/llm/llm_build
```
**Note**: TensorRT 10.7 on JetPack 6.2 lacks `DataType::kFP4` - use `#if NV_TENSORRT_MINOR >= 8` guards for FP4 code paths.

### Python Package
```bash
pip install build && python -m build --wheel --outdir dist .
pip install dist/*.whl
```

### Pre-commit (required before commits)
```bash
pip install pre-commit && pre-commit install
```
Uses: `clang-format` (C++), `yapf`/`isort` (Python), `cmake-format`, `codespell`

## C++ Code Conventions

- **Style**: Allman braces, 4-space indent, 120 char line limit
- **Naming**:
  - Files: `camelCase.cpp`, Types: `PascalCase`
  - Variables: `localVar`, Members: `mMemberVar`
  - Constants: `kCONSTANT_NAME`, Static: `sStaticVar`
- **Memory**: Use `std::unique_ptr`/`std::shared_ptr`, no raw `new`
- **Namespace**: `trt_edgellm` - close with `} // namespace trt_edgellm`
- **Comments**: Use `//!` for Doxygen, `//!<` for member docs

## Python Code Conventions

- Line length: 120 chars, formatted with `yapf`
- CLI tools defined in `pyproject.toml` under `[project.scripts]`
- Key dependencies: `torch~=2.9.1`, `transformers==4.57.1`, `nvidia-modelopt`

## Key Patterns

### Runtime Selection (mutually exclusive)
- `LlmInferenceRuntime`: Standard/multimodal inference ([llmInferenceRuntime.h](cpp/runtime/llmInferenceRuntime.h))
- `LlmInferenceSpecDecodeRuntime`: EAGLE speculative decoding ([llmInferenceSpecDecodeRuntime.h](cpp/runtime/llmInferenceSpecDecodeRuntime.h))

### Adding New Model Support
1. Add model config in `tensorrt_edgellm/llm_models/`
2. Register in export pipeline (`onnx_export/`)
3. Update engine builder if custom ops needed
4. Add test case in `tests/test_lists/`

### Export Pipeline Entry Points
```bash
tensorrt-edgellm-quantize-llm    # Quantize model (INT4/FP8)
tensorrt-edgellm-export-llm      # Export to ONNX
tensorrt-edgellm-export-visual   # Export vision encoder
```

## Testing

```bash
# Environment setup
export LLM_SDK_DIR=$(pwd)
export ONNX_DIR=/path/to/onnx
export ENGINE_DIR=/path/to/engines

# Run tests
pip install -r tests/requirements.txt
pytest --priority=l0_pipeline_a30 -v      # Pipeline tests
pytest --priority=l0_export_ampere -v     # Export tests
```

Test config files in `tests/test_lists/*.yml`

## Important Files

- [CMakeLists.txt](CMakeLists.txt) - Root build config, requires `-DTRT_PACKAGE_DIR`
- [cpp/runtime/llmInferenceRuntime.h](cpp/runtime/llmInferenceRuntime.h) - Main runtime API
- [tensorrt_edgellm/scripts/](tensorrt_edgellm/scripts/) - All Python CLI tools
- [CODING_GUIDELINES.md](CODING_GUIDELINES.md) - Full C++ style guide
- [examples/llm/llm_inference.cpp](examples/llm/llm_inference.cpp) - Reference inference example
