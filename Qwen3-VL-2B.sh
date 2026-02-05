#!/bin/bash

echo "======================================================"
echo " Jetson Orin Smart Resume Setup (이어하기 모드)"
echo "======================================================"

# 에러 발생 시 중단 (복구 구간 제외)
set +e

# ------------------------------------------------------
# [0] 스왑 메모리 확인 (가장 중요 - OOM 방지)
# ------------------------------------------------------
echo ">> [0/9] Checking Swap Memory..."
if grep -q "/swapfile" /proc/swaps; then
    echo "   ✅ Swap is already active. Skipping creation."
else
    echo "   💾 Creating/Activating 16GB Swap..."
    sudo swapoff -a
    if [ ! -f /swapfile ]; then
        sudo fallocate -l 16G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
    fi
    sudo swapon /swapfile
fi

# ------------------------------------------------------
# [1] Conda 환경 설정 (필수)
# ------------------------------------------------------
echo ">> [1/9] Loading Conda..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_test
echo "   ✅ Conda 'env_test' activated."

# ------------------------------------------------------
# [2] HF CLI 확인
# ------------------------------------------------------
echo ">> [2/9] Checking HF CLI..."
export PATH=$HOME/.local/bin:$PATH
if command -v hf &> /dev/null; then
    echo "   ✅ 'hf' command found. Skipping install."
else
    echo "   ⚠️ Installing HF CLI..."
    curl -LsSf https://hf.co/cli/install.sh | bash
fi

# ------------------------------------------------------
# [3] 패키지 복구 (빠르게 넘어감)
# ------------------------------------------------------
echo ">> [3/9] Checking APT & System..."
# APT 에러 방지용 청소는 항상 수행 (순식간임)
sudo rm -f /etc/apt/sources.list.d/cudss-local-tegra-repo-*.list
sudo rm -f /etc/apt/sources.list.d/nv-tensorrt-local-tegra-repo-*.list > /dev/null 2>&1
echo "   ✅ System clean."

# ------------------------------------------------------
# [4] Setup 레포 & 휠 설치 확인
# ------------------------------------------------------
echo ">> [4/9] Checking Setup Repo & Wheels..."
if [ -d ~/Setup ]; then
    echo "   ✅ 'Setup' folder exists. Skipping clone."
else
    echo "   📥 Cloning Setup repo..."
    cd ~
    git clone https://github.com/smyu9150-create/Setup.git
fi

# ------------------------------------------------------
# [5] Python 라이브러리 (이미 설치됐으면 pip가 알아서 스킵함)
# ------------------------------------------------------
echo ">> [5/9] Verifying Python Libraries..."
cd ~/Setup
# whl 파일이 있으면 설치 시도 (이미 깔려있으면 'Requirement already satisfied' 뜨고 1초컷)
if [ -f "torch-2.8.0-cp310-cp310-linux_aarch64.whl" ]; then
    pip install torch-2.8.0-cp310-cp310-linux_aarch64.whl > /dev/null 2>&1
fi
if [ -f "torchvision-0.23.0-cp310-cp310-manylinux_2_28_aarch64.whl" ]; then
    pip install torchvision-0.23.0-cp310-cp310-manylinux_2_28_aarch64.whl > /dev/null 2>&1
fi
pip install torch-tensorrt==2.8.0 --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps > /dev/null 2>&1
pip install dllist opencv-python requests > /dev/null 2>&1
echo "   ✅ Python libs verification done."

# ------------------------------------------------------
# [6] TensorRT 링크 확인
# ------------------------------------------------------
echo ">> [6/9] Checking TensorRT Links..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
if [ -L "$SITE_PACKAGES/tensorrt" ]; then
    echo "   ✅ TensorRT linked already. Skipping."
else
    echo "   🔗 Linking TensorRT..."
    cd $SITE_PACKAGES
    ln -sf /usr/lib/python3.10/dist-packages/tensorrt* .
    ln -sf /usr/lib/python3.10/dist-packages/graphsurgeon* .
    ln -sf /usr/lib/python3.10/dist-packages/onnx_graphsurgeon* .
    ln -sf /usr/lib/python3.10/dist-packages/uff* .
fi

# ------------------------------------------------------
# [7] 프로젝트 빌드 (이미 했으면 make가 알아서 스킵)
# ------------------------------------------------------
echo ">> [7/9] Building Orin-TensorRT-EDGE-LLM..."
cd ~
if [ ! -d "Orin-TensorRT-EDGE-LLM" ]; then
    git clone https://github.com/smyu9150-create/Orin-TensorRT-EDGE-LLM.git
fi
cd ~/Orin-TensorRT-EDGE-LLM
mkdir -p build && cd build

export LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# cmake는 다시 돌려도 안전함
cmake .. \
  -DTRT_PACKAGE_DIR=/usr \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_LIBRARY_PATH=/usr/local/cuda-12.6/lib64 \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson \
  -DCUDA_VERSION=12.6 > /dev/null 2>&1

make -j$(nproc)

# ------------------------------------------------------
# [8] 모델 다운로드 (파일 있으면 스킵)
# ------------------------------------------------------
echo ">> [8/9] Checking Model..."
cd ~/Orin-TensorRT-EDGE-LLM
mkdir -p onnx_models/qwen3-vl-2b-int4

# 핵심 파일이 있는지 검사
if [ -f "./onnx_models/qwen3-vl-2b-int4/rank0.onnx" ]; then
    echo "   ✅ Model already downloaded. Skipping."
else
    echo "   📥 Downloading Model..."
    hf download awesomesungmin/Qwen3-VL-2B-in4_AWQ --local-dir ./onnx_models/qwen3-vl-2b-int4 --exclude "*.git*"
fi

# ------------------------------------------------------
# [9] 엔진 빌드 및 실행 (여기가 문제였으므로 무조건 재시도)
# ------------------------------------------------------
echo ">> [9/9] Building Engines & Running..."
mkdir -p engines/qwen3-vl-2b-int4
mkdir -p visual_engines/qwen3-vl-2b-int4

# 🔥 메모리 확보
echo "🧹 Clearing Memory Cache for Build..."
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# LLM 엔진 빌드 (엔진 파일 없으면 빌드)
if [ -f "./engines/qwen3-vl-2b-int4/rank0.engine" ]; then
    echo "   ✅ LLM Engine already exists. Skipping build."
else
    echo "   🔨 Building LLM Engine..."
    ./build/examples/llm/llm_build \
        --onnxDir ./onnx_models/qwen3-vl-2b-int4 \
        --engineDir ./engines/qwen3-vl-2b-int4 \
        --vlm
fi

# Visual 엔진 빌드
echo "🧹 Clearing Memory Cache again..."
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

if [ -f "./visual_engines/qwen3-vl-2b-int4/visual_encoder.engine" ]; then
    echo "   ✅ Visual Engine already exists. Skipping build."
else
    echo "   🔨 Building Visual Engine..."
    ./build/examples/multimodal/visual_build \
        --onnxDir ./onnx_models/qwen3-vl-2b-int4/visual_enc_onnx \
        --engineDir ./visual_engines/qwen3-vl-2b-int4
fi

# 설정 파일 복사
cp ./onnx_models/qwen3-vl-2b-int4/preprocessor_config.json ./visual_engines/qwen3-vl-2b-int4/
cp ./onnx_models/qwen3-vl-2b-int4/video_preprocessor_config.json ./visual_engines/qwen3-vl-2b-int4/

echo "🚀 Launching Application..."
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
python integrated_qwen3-vl-2b-int4-webcam.py
