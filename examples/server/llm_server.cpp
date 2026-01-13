/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * OpenAI-compatible HTTP server for TensorRT Edge-LLM.
 * Engine is loaded once at startup and kept in memory for fast inference.
 *
 * Usage:
 *   ./llm_server --engineDir=/path/to/engine --port=8000
 *
 * Build:
 *   Add to examples/CMakeLists.txt and rebuild
 */

#include "common/trtUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <uuid/uuid.h>

using namespace trt_edgellm;
using Json = nlohmann::json;

// Global flag for graceful shutdown
std::atomic<bool> gRunning{true};

// Server configuration
struct ServerConfig
{
    std::string engineDir;
    std::string multimodalEngineDir;
    std::string modelName{"qwen3-vl"};
    int port{58010};
    bool debug{false};
};

// Runtime holder - keeps engine in memory
struct RuntimeHolder
{
    std::unique_ptr<rt::LLMInferenceRuntime> runtime;
    std::mutex mutex;
    cudaStream_t stream;
    bool initialized{false};
};

RuntimeHolder gRuntime;
ServerConfig gConfig;

// Generate UUID for request ID
std::string generateUUID()
{
    uuid_t uuid;
    uuid_generate_random(uuid);
    char str[37];
    uuid_unparse_lower(uuid, str);
    return std::string(str).substr(0, 8);
}

// Base64 decoding table
static const std::string base64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Check if a string is a base64 data URL (e.g., "data:image/png;base64,...")
bool isBase64DataUrl(std::string const& str)
{
    return str.find("data:image/") == 0 && str.find(";base64,") != std::string::npos;
}

// Decode base64 string to bytes
std::vector<unsigned char> decodeBase64(std::string const& encoded)
{
    std::vector<unsigned char> decoded;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++)
    {
        T[base64Chars[i]] = i;
    }

    int val = 0, valb = -8;
    for (unsigned char c : encoded)
    {
        if (T[c] == -1)
            continue; // Skip non-base64 characters (like whitespace)
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0)
        {
            decoded.push_back(static_cast<unsigned char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return decoded;
}

// Extract base64 data from a data URL
std::string extractBase64FromDataUrl(std::string const& dataUrl)
{
    auto pos = dataUrl.find(";base64,");
    if (pos == std::string::npos)
        return "";
    return dataUrl.substr(pos + 8); // Skip ";base64,"
}

// Maximum image dimension to avoid exceeding token limits
// VLM typically uses ~28x28 patches, so 896x896 gives about 32x32=1024 patches
constexpr int64_t kMAX_IMAGE_DIMENSION = 896;

// Resize image if it exceeds max dimension while preserving aspect ratio
rt::imageUtils::ImageData resizeImageIfNeeded(rt::imageUtils::ImageData&& image, int requestId, bool debug)
{
    int64_t maxDim = std::max(image.width, image.height);
    if (maxDim <= kMAX_IMAGE_DIMENSION)
    {
        return std::move(image);
    }

    // Calculate new dimensions preserving aspect ratio
    double scale = static_cast<double>(kMAX_IMAGE_DIMENSION) / maxDim;
    int64_t newWidth = static_cast<int64_t>(image.width * scale);
    int64_t newHeight = static_cast<int64_t>(image.height * scale);

    // Ensure dimensions are at least 1
    newWidth = std::max(newWidth, int64_t(1));
    newHeight = std::max(newHeight, int64_t(1));

    if (debug)
    {
        std::cout << "[" << requestId << "] Resizing image from " << image.width << "x" << image.height << " to "
                  << newWidth << "x" << newHeight << std::endl;
    }

    // Create output buffer
    rt::imageUtils::ImageData resized;
    resized.buffer = std::make_shared<rt::Tensor>(
        rt::Tensor({newHeight, newWidth, image.channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8,
            "resizedImage"));
    resized.width = newWidth;
    resized.height = newHeight;
    resized.channels = image.channels;

    // Perform resize
    rt::imageUtils::resizeImage(image, resized, newWidth, newHeight);

    return resized;
}

// Parse HTTP request
struct HttpRequest
{
    std::string method;
    std::string path;
    std::string body;
    std::map<std::string, std::string> headers;
};

HttpRequest parseHttpRequest(std::string const& raw)
{
    HttpRequest req;
    std::istringstream stream(raw);
    std::string line;

    // Parse request line
    std::getline(stream, line);
    std::istringstream requestLine(line);
    requestLine >> req.method >> req.path;

    // Parse headers
    while (std::getline(stream, line) && line != "\r" && !line.empty())
    {
        if (line.back() == '\r')
            line.pop_back();
        auto colonPos = line.find(':');
        if (colonPos != std::string::npos)
        {
            std::string key = line.substr(0, colonPos);
            std::string value = line.substr(colonPos + 2);
            req.headers[key] = value;
        }
    }

    // Parse body (rest of the content)
    std::ostringstream bodyStream;
    bodyStream << stream.rdbuf();
    req.body = bodyStream.str();

    return req;
}

// Send all data reliably
bool sendAll(int clientSocket, char const* data, size_t length)
{
    size_t totalSent = 0;
    while (totalSent < length)
    {
        ssize_t sent = send(clientSocket, data + totalSent, length - totalSent, 0);
        if (sent <= 0)
        {
            return false;
        }
        totalSent += sent;
    }
    return true;
}

// Send HTTP response
void sendResponse(int clientSocket, int statusCode, std::string const& statusText, std::string const& contentType,
    std::string const& body)
{
    std::ostringstream response;
    response << "HTTP/1.1 " << statusCode << " " << statusText << "\r\n";
    response << "Content-Type: " << contentType << "; charset=utf-8\r\n";
    response << "Content-Length: " << body.length() << "\r\n";
    response << "Access-Control-Allow-Origin: *\r\n";
    response << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    response << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";
    response << "Connection: close\r\n";
    response << "\r\n";
    response << body;

    std::string responseStr = response.str();
    sendAll(clientSocket, responseStr.c_str(), responseStr.length());
}

// Handle health check
void handleHealth(int clientSocket)
{
    Json response;
    response["status"] = "healthy";
    response["model"] = gConfig.modelName;
    response["engine_loaded"] = gRuntime.initialized;
    sendResponse(clientSocket, 200, "OK", "application/json", response.dump());
}

// Handle models list
void handleModels(int clientSocket)
{
    Json response;
    response["object"] = "list";
    response["data"] = Json::array();

    Json model;
    model["id"] = gConfig.modelName;
    model["object"] = "model";
    model["created"] = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    model["owned_by"] = "tensorrt-edgellm";
    response["data"].push_back(model);

    sendResponse(clientSocket, 200, "OK", "application/json", response.dump());
}

// Handle chat completions
void handleChatCompletions(int clientSocket, std::string const& body, std::string const& clientAddr)
{
    auto requestStartTime = std::chrono::high_resolution_clock::now();
    static std::atomic<int> requestCounter{0};
    int requestId = ++requestCounter;

    std::cout << "\n========== Request #" << requestId << " ==========" << std::endl;
    std::cout << "[" << requestId << "] Client: " << clientAddr << std::endl;
    std::cout << "[" << requestId << "] Time: " << std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << std::endl;

    try
    {
        Json request = Json::parse(body);

        // Log raw request for debugging
        if (gConfig.debug)
        {
            std::cout << "[" << requestId << "] Raw request: " << body.substr(0, std::min(body.length(), (size_t)500)) 
                      << (body.length() > 500 ? "..." : "") << std::endl;
        }

        // Extract parameters (with sensible defaults)
        float temperature = request.value("temperature", 0.7f);
        float topP = request.value("top_p", 0.9f);
        int64_t topK = request.value("top_k", 50);
        int64_t maxTokens = request.value("max_tokens", 2048);  // Default 2048 for longer responses
        bool streamMode = request.value("stream", false);
        auto messages = request["messages"];

        // Log request info with indicators for default values
        std::cout << "[" << requestId << "] Model: " << request.value("model", gConfig.modelName) << std::endl;
        std::cout << "[" << requestId << "] Parameters: ";
        std::cout << "temp=" << temperature << (request.contains("temperature") ? "" : "(default)") << ", ";
        std::cout << "topP=" << topP << (request.contains("top_p") ? "" : "(default)") << ", ";
        std::cout << "topK=" << topK << (request.contains("top_k") ? "" : "(default)") << ", ";
        std::cout << "maxTokens=" << maxTokens << (request.contains("max_tokens") ? "" : "(default)") << std::endl;
        std::cout << "[" << requestId << "] Stream: " << (streamMode ? "true" : "false") << std::endl;
        std::cout << "[" << requestId << "] Messages count: " << messages.size() << std::endl;
        
        // Log each message (truncated)
        for (size_t i = 0; i < messages.size(); i++)
        {
            std::string role = messages[i]["role"].get<std::string>();
            std::string content;
            if (messages[i]["content"].is_string())
            {
                content = messages[i]["content"].get<std::string>();
            }
            else
            {
                content = "[multimodal content]";
            }
            // Truncate long content
            if (content.length() > 100)
            {
                content = content.substr(0, 100) + "...";
            }
            std::cout << "[" << requestId << "] Message[" << i << "] " << role << ": " << content << std::endl;
        }

        // Convert to Edge-LLM format
        std::vector<rt::Message> llmMessages;
        std::vector<rt::imageUtils::ImageData> imageBuffers;

        for (auto const& msg : messages)
        {
            rt::Message llmMsg;
            llmMsg.role = msg["role"].get<std::string>();

            if (msg["content"].is_string())
            {
                // Simple text content
                rt::Message::MessageContent msgContent;
                msgContent.type = "text";
                msgContent.content = msg["content"].get<std::string>();
                llmMsg.contents.push_back(msgContent);
            }
            else if (msg["content"].is_array())
            {
                // Array of content items
                for (auto const& contentItem : msg["content"])
                {
                    rt::Message::MessageContent msgContent;
                    std::string type = contentItem.value("type", "text");

                    if (type == "text")
                    {
                        msgContent.type = "text";
                        msgContent.content = contentItem.value("text", "");
                    }
                    else if (type == "image" || type == "image_url")
                    {
                        msgContent.type = "image";
                        std::string imagePath;
                        if (contentItem.contains("image"))
                        {
                            imagePath = contentItem["image"].get<std::string>();
                        }
                        else if (contentItem.contains("image_url"))
                        {
                            auto imageUrl = contentItem["image_url"];
                            if (imageUrl.is_string())
                            {
                                imagePath = imageUrl.get<std::string>();
                            }
                            else
                            {
                                imagePath = imageUrl.value("url", "");
                            }
                        }
                        msgContent.content = imagePath;

                        // Load image based on source type
                        if (!imagePath.empty())
                        {
                            if (isBase64DataUrl(imagePath))
                            {
                                // Decode base64 data URL and load from memory
                                std::string base64Data = extractBase64FromDataUrl(imagePath);
                                if (!base64Data.empty())
                                {
                                    auto decodedData = decodeBase64(base64Data);
                                    if (!decodedData.empty())
                                    {
                                        auto image = rt::imageUtils::loadImageFromMemory(
                                            decodedData.data(), decodedData.size());
                                        if (image.buffer != nullptr)
                                        {
                                            if (gConfig.debug)
                                            {
                                                std::cout << "[" << requestId << "] Loaded base64 image: "
                                                          << decodedData.size() << " bytes -> " << image.width << "x"
                                                          << image.height << "x" << image.channels << std::endl;
                                            }
                                            // Resize if too large
                                            image = resizeImageIfNeeded(std::move(image), requestId, gConfig.debug);
                                            imageBuffers.push_back(std::move(image));
                                            // Update content to indicate image was loaded
                                            msgContent.content = "<image_loaded>";
                                        }
                                        else
                                        {
                                            std::cerr << "[" << requestId
                                                      << "] Failed to decode base64 image data" << std::endl;
                                        }
                                    }
                                }
                            }
                            else if (imagePath.find("http") != 0)
                            {
                                // Load from file path
                                auto image = rt::imageUtils::loadImageFromFile(imagePath);
                                if (image.buffer != nullptr)
                                {
                                    // Resize if too large
                                    image = resizeImageIfNeeded(std::move(image), requestId, gConfig.debug);
                                    imageBuffers.push_back(std::move(image));
                                }
                            }
                            // HTTP URLs are not supported yet
                        }
                    }
                    llmMsg.contents.push_back(msgContent);
                }
            }
            llmMessages.push_back(llmMsg);
        }

        // Build generation request
        rt::LLMGenerationRequest genRequest;
        rt::LLMGenerationRequest::Request req;
        req.messages = std::move(llmMessages);
        req.imageBuffers = std::move(imageBuffers);
        genRequest.requests.push_back(std::move(req));

        genRequest.temperature = temperature;
        genRequest.topP = topP;
        genRequest.topK = topK;
        genRequest.maxGenerateLength = maxTokens;
        genRequest.applyChatTemplate = true;
        genRequest.addGenerationPrompt = true;

        std::string responseId = "chatcmpl-" + generateUUID();

        // Estimate prompt tokens
        int promptTokens = 0;
        for (auto const& msg : messages)
        {
            if (msg["content"].is_string())
            {
                promptTokens += msg["content"].get<std::string>().length() / 4;
            }
        }

        if (streamMode)
        {
            // TRUE STREAMING: Send SSE header first, then stream tokens as they're generated
            std::string sseHeader = "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream; charset=utf-8\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: keep-alive\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                "\r\n";
            
            sendAll(clientSocket, sseHeader.c_str(), sseHeader.length());

            // Send role first
            Json roleChunk;
            roleChunk["id"] = responseId;
            roleChunk["object"] = "chat.completion.chunk";
            roleChunk["created"] = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            roleChunk["model"] = request.value("model", gConfig.modelName);
            Json roleDelta;
            roleDelta["role"] = "assistant";
            Json roleChoice;
            roleChoice["index"] = 0;
            roleChoice["delta"] = roleDelta;
            roleChoice["finish_reason"] = nullptr;
            roleChunk["choices"] = Json::array({roleChoice});
            std::string roleData = "data: " + roleChunk.dump() + "\n\n";
            sendAll(clientSocket, roleData.c_str(), roleData.length());

            // Streaming token callback
            auto inferenceStartTime = std::chrono::high_resolution_clock::now();
            std::chrono::time_point<std::chrono::high_resolution_clock> firstTokenTime;
            int tokenCount = 0;
            std::string fullOutput;
            std::string modelName = request.value("model", gConfig.modelName);

            auto tokenCallback = [&](std::string const& token, bool isFirst) -> bool {
                if (isFirst)
                {
                    firstTokenTime = std::chrono::high_resolution_clock::now();
                }
                tokenCount++;
                fullOutput += token;

                // Send token via SSE
                Json chunk;
                chunk["id"] = responseId;
                chunk["object"] = "chat.completion.chunk";
                chunk["created"] = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                chunk["model"] = modelName;
                
                Json delta;
                delta["content"] = token;
                
                Json choice;
                choice["index"] = 0;
                choice["delta"] = delta;
                choice["finish_reason"] = nullptr;
                chunk["choices"] = Json::array({choice});

                std::string sseData = "data: " + chunk.dump() + "\n\n";
                return sendAll(clientSocket, sseData.c_str(), sseData.length());
            };

            // Run streaming inference
            rt::LLMGenerationResponse genResponse;
            bool success;
            {
                std::lock_guard<std::mutex> lock(gRuntime.mutex);
                success = gRuntime.runtime->handleRequestStreaming(genRequest, genResponse, gRuntime.stream, tokenCallback);
            }

            auto inferenceEndTime = std::chrono::high_resolution_clock::now();

            if (!success)
            {
                throw std::runtime_error("Streaming inference failed");
            }

            // Send finish chunk
            Json finishChunk;
            finishChunk["id"] = responseId;
            finishChunk["object"] = "chat.completion.chunk";
            finishChunk["created"] = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            finishChunk["model"] = modelName;
            
            Json finishDelta;
            Json finishChoice;
            finishChoice["index"] = 0;
            finishChoice["delta"] = finishDelta;
            finishChoice["finish_reason"] = "stop";
            finishChunk["choices"] = Json::array({finishChoice});

            std::string finishData = "data: " + finishChunk.dump() + "\n\n";
            sendAll(clientSocket, finishData.c_str(), finishData.length());

            // Send [DONE] marker
            std::string doneData = "data: [DONE]\n\n";
            sendAll(clientSocket, doneData.c_str(), doneData.length());

            // Log performance metrics
            auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                inferenceEndTime - inferenceStartTime).count();
            auto ttftMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                firstTokenTime - inferenceStartTime).count();
            double tokensPerSecond = (totalMs > 0 && tokenCount > 0) 
                ? (tokenCount * 1000.0 / totalMs) : 0.0;

            std::cout << "[" << requestId << "] Inference time: " << totalMs << " ms" << std::endl;
            std::cout << "[" << requestId << "] Output tokens: " << tokenCount << std::endl;
            std::cout << "[" << requestId << "] TTFT: " << ttftMs << " ms" << std::endl;
            std::cout << "[" << requestId << "] Throughput: " << std::fixed << std::setprecision(2) << tokensPerSecond << " tokens/s" << std::endl;
            
            std::string outputPreview = fullOutput;
            if (outputPreview.length() > 200)
            {
                outputPreview = outputPreview.substr(0, 200) + "...";
            }
            std::cout << "[" << requestId << "] Response: " << outputPreview << std::endl;
            std::cout << "[" << requestId << "] Response ID: " << responseId << std::endl;
            std::cout << "[" << requestId << "] Response mode: TRUE STREAMING" << std::endl;
        }
        else
        {
            // Non-streaming mode
            auto inferenceStartTime = std::chrono::high_resolution_clock::now();
            rt::LLMGenerationResponse genResponse;
            {
                std::lock_guard<std::mutex> lock(gRuntime.mutex);
                if (!gRuntime.runtime->handleRequest(genRequest, genResponse, gRuntime.stream))
                {
                    throw std::runtime_error("Inference failed");
                }
            }

            auto inferenceEndTime = std::chrono::high_resolution_clock::now();
            auto inferenceMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                inferenceEndTime - inferenceStartTime).count();

            std::string outputText = genResponse.outputTexts.empty() ? "" : genResponse.outputTexts[0];
            int outputTokenCount = genResponse.outputIds.empty() ? 0 : genResponse.outputIds[0].size();

            double tokensPerSecond = (inferenceMs > 0 && outputTokenCount > 0) 
                ? (outputTokenCount * 1000.0 / inferenceMs) : 0.0;

            std::cout << "[" << requestId << "] Inference time: " << inferenceMs << " ms" << std::endl;
            std::cout << "[" << requestId << "] Output tokens: " << outputTokenCount << std::endl;
            std::cout << "[" << requestId << "] Throughput: " << std::fixed << std::setprecision(2) << tokensPerSecond << " tokens/s" << std::endl;
            
            std::string outputPreview = outputText;
            if (outputPreview.length() > 200)
            {
                outputPreview = outputPreview.substr(0, 200) + "...";
            }
            std::cout << "[" << requestId << "] Response: " << outputPreview << std::endl;

            int completionTokens = outputTokenCount > 0 ? outputTokenCount : outputText.length() / 4;

            Json response;
            response["id"] = responseId;
            response["object"] = "chat.completion";
            response["created"] = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            response["model"] = request.value("model", gConfig.modelName);

            Json choice;
            choice["index"] = 0;
            choice["message"]["role"] = "assistant";
            choice["message"]["content"] = outputText;
            choice["finish_reason"] = "stop";
            response["choices"] = Json::array({choice});

            response["usage"]["prompt_tokens"] = promptTokens;
            response["usage"]["completion_tokens"] = completionTokens;
            response["usage"]["total_tokens"] = promptTokens + completionTokens;

            std::string responseBody = response.dump();
            
            std::cout << "[" << requestId << "] Response ID: " << responseId << std::endl;
            std::cout << "[" << requestId << "] Response mode: NON-STREAMING" << std::endl;
            
            sendResponse(clientSocket, 200, "OK", "application/json", responseBody);
        }
        
        auto totalEndTime = std::chrono::high_resolution_clock::now();
        auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            totalEndTime - requestStartTime).count();
        std::cout << "[" << requestId << "] Total time: " << totalMs << " ms" << std::endl;
        std::cout << "[" << requestId << "] Status: SUCCESS" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
    catch (std::exception const& e)
    {
        std::cout << "[" << requestId << "] ERROR: " << e.what() << std::endl;
        std::cout << "[" << requestId << "] Status: FAILED" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        Json error;
        error["error"]["message"] = std::string("Error: ") + e.what();
        error["error"]["type"] = "server_error";
        sendResponse(clientSocket, 500, "Internal Server Error", "application/json", error.dump());
    }
}

// Handle client connection
void handleClient(int clientSocket, std::string clientAddr)
{
    // Set socket timeout
    struct timeval tv;
    tv.tv_sec = 30;
    tv.tv_usec = 0;
    setsockopt(clientSocket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // Read HTTP headers first
    std::string rawRequest;
    char buffer[4096];
    size_t headerEnd = std::string::npos;

    while (headerEnd == std::string::npos)
    {
        ssize_t bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
        if (bytesRead <= 0)
        {
            close(clientSocket);
            return;
        }
        buffer[bytesRead] = '\0';
        rawRequest.append(buffer, bytesRead);
        headerEnd = rawRequest.find("\r\n\r\n");
    }

    // Parse Content-Length from headers
    size_t contentLength = 0;
    std::string headerPart = rawRequest.substr(0, headerEnd);
    size_t clPos = headerPart.find("Content-Length:");
    if (clPos == std::string::npos)
    {
        clPos = headerPart.find("content-length:");
    }
    if (clPos != std::string::npos)
    {
        size_t valueStart = clPos + 15;
        while (valueStart < headerPart.length() && (headerPart[valueStart] == ' ' || headerPart[valueStart] == ':'))
        {
            valueStart++;
        }
        size_t valueEnd = headerPart.find("\r\n", valueStart);
        if (valueEnd != std::string::npos)
        {
            contentLength = std::stoul(headerPart.substr(valueStart, valueEnd - valueStart));
        }
    }

    // Read remaining body if needed
    size_t bodyStart = headerEnd + 4;
    size_t bodyReceived = rawRequest.length() - bodyStart;
    while (bodyReceived < contentLength)
    {
        ssize_t bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
        if (bytesRead <= 0)
        {
            break;
        }
        buffer[bytesRead] = '\0';
        rawRequest.append(buffer, bytesRead);
        bodyReceived += bytesRead;
    }

    HttpRequest req = parseHttpRequest(rawRequest);

    if (gConfig.debug)
    {
        std::cout << "[DEBUG] " << req.method << " " << req.path << std::endl;
    }

    // Handle CORS preflight
    if (req.method == "OPTIONS")
    {
        sendResponse(clientSocket, 200, "OK", "text/plain", "");
    }
    // Route requests
    else if (req.path == "/health" && req.method == "GET")
    {
        handleHealth(clientSocket);
    }
    else if (req.path == "/v1/models" && req.method == "GET")
    {
        handleModels(clientSocket);
    }
    else if (req.path == "/v1/chat/completions" && req.method == "POST")
    {
        handleChatCompletions(clientSocket, req.body, clientAddr);
    }
    else
    {
        Json error;
        error["error"]["message"] = "Not found";
        error["error"]["type"] = "invalid_request_error";
        sendResponse(clientSocket, 404, "Not Found", "application/json", error.dump());
    }

    close(clientSocket);
}

void signalHandler(int signal)
{
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    gRunning = false;
}

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " [options]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --engineDir <path>        Path to LLM engine directory (required)" << std::endl;
    std::cerr << "  --multimodalEngineDir <path>  Path to visual encoder engine directory" << std::endl;
    std::cerr << "  --modelName <name>        Model name to report (default: qwen3-vl)" << std::endl;
    std::cerr << "  --port <port>             HTTP port (default: 8000)" << std::endl;
    std::cerr << "  --debug                   Enable debug logging" << std::endl;
}

int main(int argc, char* argv[])
{
    // Parse arguments
    static struct option options[] = {{"help", no_argument, 0, 'h'}, {"engineDir", required_argument, 0, 'e'},
        {"multimodalEngineDir", required_argument, 0, 'm'}, {"modelName", required_argument, 0, 'n'},
        {"port", required_argument, 0, 'p'}, {"debug", no_argument, 0, 'd'}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "he:m:n:p:d", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'h': printUsage(argv[0]); return 0;
        case 'e': gConfig.engineDir = optarg; break;
        case 'm': gConfig.multimodalEngineDir = optarg; break;
        case 'n': gConfig.modelName = optarg; break;
        case 'p': gConfig.port = std::stoi(optarg); break;
        case 'd': gConfig.debug = true; break;
        default: printUsage(argv[0]); return 1;
        }
    }

    if (gConfig.engineDir.empty())
    {
        std::cerr << "Error: --engineDir is required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Setup signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // Load TensorRT plugin
    auto pluginHandles = loadEdgellmPluginLib();

    // Initialize CUDA stream
    CUDA_CHECK(cudaStreamCreate(&gRuntime.stream));

    std::cout << "============================================================" << std::endl;
    std::cout << "TensorRT Edge-LLM OpenAI API Server (C++)" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Engine Dir: " << gConfig.engineDir << std::endl;
    std::cout << "Visual Engine Dir: " << (gConfig.multimodalEngineDir.empty() ? "None" : gConfig.multimodalEngineDir)
              << std::endl;
    std::cout << "Model Name: " << gConfig.modelName << std::endl;
    std::cout << "Port: " << gConfig.port << std::endl;
    std::cout << "============================================================" << std::endl;

    // Initialize runtime (load engine - this takes time but only once!)
    std::cout << "Loading TensorRT engine (this may take a minute)..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    try
    {
        std::unordered_map<std::string, std::string> loraWeightsMap;
        gRuntime.runtime = std::make_unique<rt::LLMInferenceRuntime>(
            gConfig.engineDir, gConfig.multimodalEngineDir, loraWeightsMap, gRuntime.stream);
        gRuntime.runtime->captureDecodingCUDAGraph(gRuntime.stream);
        gRuntime.initialized = true;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Failed to initialize runtime: " << e.what() << std::endl;
        return 1;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    std::cout << "Engine loaded successfully in " << duration << " seconds!" << std::endl;
    std::cout << "============================================================" << std::endl;

    // Create server socket
    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0)
    {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }

    // Allow address reuse
    int opt_val = 1;
    setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val));

    // Bind to port
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(gConfig.port);

    if (bind(serverSocket, (struct sockaddr*) &serverAddr, sizeof(serverAddr)) < 0)
    {
        std::cerr << "Failed to bind to port " << gConfig.port << std::endl;
        close(serverSocket);
        return 1;
    }

    // Listen
    if (listen(serverSocket, 10) < 0)
    {
        std::cerr << "Failed to listen" << std::endl;
        close(serverSocket);
        return 1;
    }

    std::cout << "Server listening on http://0.0.0.0:" << gConfig.port << std::endl;
    std::cout << "API Docs: http://0.0.0.0:" << gConfig.port << "/v1/models" << std::endl;
    std::cout << "Press Ctrl+C to stop" << std::endl;
    std::cout << "============================================================" << std::endl;

    // Accept connections
    while (gRunning)
    {
        struct sockaddr_in clientAddr;
        socklen_t clientLen = sizeof(clientAddr);

        // Use select for timeout to allow checking gRunning flag
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(serverSocket, &readfds);

        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        int activity = select(serverSocket + 1, &readfds, nullptr, nullptr, &timeout);
        if (activity < 0 && errno != EINTR)
        {
            std::cerr << "Select error" << std::endl;
            break;
        }

        if (activity > 0 && FD_ISSET(serverSocket, &readfds))
        {
            int clientSocket = accept(serverSocket, (struct sockaddr*) &clientAddr, &clientLen);
            if (clientSocket >= 0)
            {
                // Get client IP address
                char clientIp[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, &clientAddr.sin_addr, clientIp, INET_ADDRSTRLEN);
                std::string clientAddrStr = std::string(clientIp) + ":" + std::to_string(ntohs(clientAddr.sin_port));
                
                // Handle in separate thread for concurrency
                std::thread(handleClient, clientSocket, clientAddrStr).detach();
            }
        }
    }

    // Cleanup
    close(serverSocket);
    CUDA_CHECK(cudaStreamDestroy(gRuntime.stream));
    std::cout << "Server stopped." << std::endl;

    return 0;
}
