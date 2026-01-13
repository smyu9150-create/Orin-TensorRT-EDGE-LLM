#!/bin/bash
# TensorRT Edge-LLM Server Startup Script
# Start both C++ inference backend and Python API server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../../build"

# Default paths (can be overridden by environment variables)
ENGINE_DIR="${ENGINE_DIR:-/home/rm01/models/dev/trt-edge-llm-engine/llm/Qwen3-VL-8B-Instruct}"
VISUAL_ENGINE_DIR="${VISUAL_ENGINE_DIR:-/home/rm01/models/dev/trt-edge-llm-engine/visual/Qwen3-VL-8B-Instruct}"
SOCKET_PATH="${SOCKET_PATH:-/tmp/edgellm.sock}"
API_PORT="${API_PORT:-58010}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl-8b}"

# Plugin path
export EDGELLM_PLUGIN_PATH="${BUILD_DIR}/libNvInfer_edgellm_plugin.so"

echo "============================================================"
echo "TensorRT Edge-LLM Server"
echo "============================================================"
echo "Engine Dir: ${ENGINE_DIR}"
echo "Visual Dir: ${VISUAL_ENGINE_DIR}"
echo "Socket: ${SOCKET_PATH}"
echo "API Port: ${API_PORT}"
echo "Model Name: ${MODEL_NAME}"
echo "============================================================"

# Cleanup function
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $API_PID 2>/dev/null || true
    rm -f "${SOCKET_PATH}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start C++ inference backend
echo "Starting C++ inference backend..."
"${BUILD_DIR}/examples/server/llm_inference_server" \
    --engineDir="${ENGINE_DIR}" \
    --multimodalEngineDir="${VISUAL_ENGINE_DIR}" \
    --socketPath="${SOCKET_PATH}" &
BACKEND_PID=$!

# Wait for socket to be created
echo "Waiting for backend to initialize..."
for i in {1..60}; do
    if [ -S "${SOCKET_PATH}" ]; then
        echo "Backend ready!"
        break
    fi
    sleep 1
done

if [ ! -S "${SOCKET_PATH}" ]; then
    echo "Error: Backend failed to start"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start Python API server
echo "Starting Python API server..."
python3 "${SCRIPT_DIR}/openai_api_server.py" \
    --host=0.0.0.0 \
    --port="${API_PORT}" \
    --socket="${SOCKET_PATH}" \
    --model-name="${MODEL_NAME}" &
API_PID=$!

echo "============================================================"
echo "Services started!"
echo "API available at: http://0.0.0.0:${API_PORT}"
echo "============================================================"
echo "Press Ctrl+C to stop"

# Wait for either process to exit
wait
