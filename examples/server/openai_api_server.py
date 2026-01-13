#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
TensorRT Edge-LLM OpenAI-Compatible API Server

This Python server provides an OpenAI-compatible REST API that communicates
with the C++ inference backend via Unix Domain Socket.

Architecture:
  [Client] <-- HTTP/REST --> [Python FastAPI] <-- Unix Socket --> [C++ Inference Engine]

Usage:
  1. Start C++ backend:
     ./llm_inference_server --engineDir=/path/to/engine

  2. Start Python API server:
     python openai_api_server.py --port=58010

  3. Call API:
     curl http://localhost:58010/v1/chat/completions -d '{"messages": [...]}'
"""

import argparse
import json
import socket
import struct
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================================
# Configuration
# ============================================================================
class Config:
    socket_path: str = "/tmp/edgellm.sock"
    model_name: str = "qwen3-vl-8b"
    timeout: float = 120.0  # seconds


config = Config()


# ============================================================================
# Unix Socket Client
# ============================================================================
class InferenceClient:
    """Client to communicate with C++ inference server via Unix Socket."""

    def __init__(self, socket_path: str, timeout: float = 120.0):
        self.socket_path = socket_path
        self.timeout = timeout

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to C++ server and receive response."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)

        try:
            sock.connect(self.socket_path)

            # Send message with length prefix
            message = json.dumps(request).encode('utf-8')
            sock.sendall(struct.pack('!I', len(message)))
            sock.sendall(message)

            # Receive response length
            length_data = sock.recv(4)
            if len(length_data) < 4:
                raise RuntimeError("Failed to receive response length")
            msg_len = struct.unpack('!I', length_data)[0]

            # Receive response
            response_data = b''
            while len(response_data) < msg_len:
                chunk = sock.recv(min(4096, msg_len - len(response_data)))
                if not chunk:
                    break
                response_data += chunk

            return json.loads(response_data.decode('utf-8'))

        except socket.error as e:
            raise RuntimeError(f"Socket error: {e}. Is the C++ inference server running?")
        finally:
            sock.close()

    def health_check(self) -> Dict[str, Any]:
        """Check if inference server is healthy."""
        return self._send_request({"command": "health"})

    def inference(
        self,
        messages: List[Dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 50,
        enable_thinking: bool = False,
    ) -> Dict[str, Any]:
        """Run inference."""
        request = {
            "command": "inference",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "enable_thinking": enable_thinking,
        }
        return self._send_request(request)


# Global client instance
inference_client: Optional[InferenceClient] = None


# ============================================================================
# Pydantic Models (OpenAI API Compatible)
# ============================================================================
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="qwen3-vl-8b")
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    stream: bool = Field(default=False)
    enable_thinking: bool = Field(default=False)

    # Additional OpenAI fields (accepted but may not all be used)
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "tensorrt-edgellm"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    model: str
    engine_loaded: bool
    backend: str = "tensorrt-edgellm"


# ============================================================================
# FastAPI Application
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global inference_client
    inference_client = InferenceClient(config.socket_path, config.timeout)
    print(f"Initialized inference client (socket: {config.socket_path})")
    yield
    print("Shutting down...")


app = FastAPI(
    title="TensorRT Edge-LLM API",
    description="OpenAI-compatible API for TensorRT Edge-LLM",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        result = inference_client.health_check()
        return HealthResponse(
            status=result.get("status", "unknown"),
            model=config.model_name,
            engine_loaded=result.get("engine_loaded", False),
        )
    except Exception as e:
        return HealthResponse(
            status=f"error: {str(e)}",
            model=config.model_name,
            engine_loaded=False,
        )


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models (OpenAI compatible)."""
    return ModelListResponse(data=[
        ModelInfo(
            id=config.model_name,
            created=int(time.time()),
        )
    ])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion (OpenAI compatible)."""
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not yet implemented")

    try:
        # Convert messages to dict format
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Call inference backend
        start_time = time.time()
        result = inference_client.inference(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            enable_thinking=request.enable_thinking,
        )
        inference_time = time.time() - start_time

        # Check for errors
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        # Build response
        output_text = result.get("text", "")
        output_tokens = result.get("tokens", 0)

        # Estimate prompt tokens (rough)
        prompt_tokens = sum(
            len(str(m.content)) // 4 for m in request.messages
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=output_text),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=output_tokens if output_tokens > 0 else len(output_text) // 4,
                total_tokens=prompt_tokens + (output_tokens if output_tokens > 0 else len(output_text) // 4),
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# Legacy endpoint for compatibility
@app.post("/chat/completions")
async def create_chat_completion_legacy(request: ChatCompletionRequest):
    """Legacy endpoint without /v1 prefix."""
    return await create_chat_completion(request)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="TensorRT Edge-LLM OpenAI API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=58010, help="Port to bind")
    parser.add_argument("--socket", type=str, default="/tmp/edgellm.sock", help="Unix socket path for C++ backend")
    parser.add_argument("--model-name", type=str, default="qwen3-vl-8b", help="Model name to report")
    parser.add_argument("--timeout", type=float, default=120.0, help="Inference timeout in seconds")
    args = parser.parse_args()

    config.socket_path = args.socket
    config.model_name = args.model_name
    config.timeout = args.timeout

    print("=" * 60)
    print("TensorRT Edge-LLM OpenAI API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Backend Socket: {config.socket_path}")
    print(f"Model Name: {config.model_name}")
    print("=" * 60)
    print("Make sure C++ inference server is running:")
    print(f"  ./llm_inference_server --engineDir=<path> --socketPath={config.socket_path}")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
