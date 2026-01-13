#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible API server for TensorRT Edge-LLM.

Usage:
    python openai_server.py \
        --engine-dir /path/to/llm/engine \
        --visual-engine-dir /path/to/visual/engine \
        --host 0.0.0.0 --port 8000

API Endpoints:
    POST /v1/chat/completions - OpenAI-compatible chat completions
    GET /v1/models - List available models
    GET /health - Health check
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============== Pydantic Models ==============

class ChatMessage(BaseModel):
    role: str
    content: str | list


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-vl"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stream: bool = False  # TODO: implement streaming


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "tensorrt-edgellm"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ============== Server Configuration ==============

class ServerConfig:
    def __init__(self):
        self.engine_dir: str = ""
        self.visual_engine_dir: Optional[str] = None
        self.llm_inference_bin: str = ""
        self.model_name: str = "qwen3-vl"


config = ServerConfig()
app = FastAPI(
    title="TensorRT Edge-LLM OpenAI API",
    description="OpenAI-compatible API server for TensorRT Edge-LLM inference",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Helper Functions ==============

def convert_to_edgellm_format(request: ChatCompletionRequest) -> dict:
    """Convert OpenAI format to TensorRT Edge-LLM input format."""
    messages = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
        else:
            # Handle multimodal content (list format)
            content_list = []
            for item in msg.content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        content_list.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        # Convert OpenAI image_url format to Edge-LLM format
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
                        # Handle base64 or file path
                        if url.startswith("data:"):
                            # TODO: save base64 to temp file
                            pass
                        else:
                            content_list.append({"type": "image", "image": url})
            messages.append({"role": msg.role, "content": content_list if content_list else msg.content})

    return {
        "batch_size": 1,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "max_generate_length": request.max_tokens,
        "requests": [{"messages": messages}]
    }


def run_inference(input_data: dict) -> str:
    """Run TensorRT Edge-LLM inference and return the response."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
        json.dump(input_data, input_file)
        input_path = input_file.name

    output_path = input_path.replace('.json', '_output.json')

    try:
        cmd = [
            config.llm_inference_bin,
            f"--engineDir={config.engine_dir}",
            f"--inputFile={input_path}",
            f"--outputFile={output_path}",
        ]

        if config.visual_engine_dir:
            cmd.append(f"--multimodalEngineDir={config.visual_engine_dir}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(f"Inference failed: {result.stderr}")

        with open(output_path, 'r') as f:
            output_data = json.load(f)

        # Extract response text - TensorRT Edge-LLM format
        if "responses" in output_data and len(output_data["responses"]) > 0:
            return output_data["responses"][0].get("output_text", "")
        elif "results" in output_data and len(output_data["results"]) > 0:
            return output_data["results"][0].get("output_text", "")
        else:
            # Try to find any response field
            return str(output_data)

    finally:
        # Cleanup temp files
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


# ============== API Endpoints ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": config.model_name}


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models (OpenAI-compatible)."""
    return ModelListResponse(
        data=[
            ModelInfo(
                id=config.model_name,
                created=int(time.time())
            )
        ]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not yet implemented")

    try:
        # Convert request format
        edgellm_input = convert_to_edgellm_format(request)

        # Run inference
        start_time = time.time()
        response_text = run_inference(edgellm_input)
        inference_time = time.time() - start_time

        # Estimate token counts (rough approximation)
        prompt_tokens = sum(len(str(m.content).split()) for m in request.messages) * 2
        completion_tokens = len(response_text.split()) * 2

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Inference timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible API server for TensorRT Edge-LLM")
    parser.add_argument("--engine-dir", required=True, help="Path to LLM engine directory")
    parser.add_argument("--visual-engine-dir", default=None, help="Path to visual encoder engine directory (for VLM)")
    parser.add_argument("--llm-inference-bin", default=None, help="Path to llm_inference binary")
    parser.add_argument("--model-name", default="qwen3-vl", help="Model name to report")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")

    args = parser.parse_args()

    # Configure server
    config.engine_dir = args.engine_dir
    config.visual_engine_dir = args.visual_engine_dir
    config.model_name = args.model_name

    # Find llm_inference binary
    if args.llm_inference_bin:
        config.llm_inference_bin = args.llm_inference_bin
    else:
        # Try common locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "../../build/examples/llm/llm_inference"),
            "/home/rm01/TensorRT-Edge-LLM/build/examples/llm/llm_inference",
            "llm_inference",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config.llm_inference_bin = os.path.abspath(path)
                break
        else:
            raise FileNotFoundError("Could not find llm_inference binary. Use --llm-inference-bin to specify.")

    print(f"=" * 60)
    print(f"TensorRT Edge-LLM OpenAI API Server")
    print(f"=" * 60)
    print(f"Engine Dir: {config.engine_dir}")
    print(f"Visual Engine Dir: {config.visual_engine_dir or 'None (text-only)'}")
    print(f"LLM Inference Binary: {config.llm_inference_bin}")
    print(f"Model Name: {config.model_name}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print(f"=" * 60)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
