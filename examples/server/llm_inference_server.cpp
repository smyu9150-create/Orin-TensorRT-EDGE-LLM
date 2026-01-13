/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * TensorRT Edge-LLM Inference Server (Unix Socket)
 * Engine is loaded once at startup and kept in memory for fast inference.
 * Communicates with Python API layer via Unix Domain Socket.
 *
 * Protocol: JSON over Unix Socket
 * Request:  {"messages": [...], "max_tokens": N, "temperature": T, ...}
 * Response: {"text": "...", "tokens": N, "error": null}
 */

#include "common/trtUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <arpa/inet.h>

using namespace trt_edgellm;
using Json = nlohmann::json;

// Global flag for graceful shutdown
std::atomic<bool> gRunning{true};

// Server configuration
struct ServerConfig
{
    std::string engineDir;
    std::string multimodalEngineDir;
    std::string socketPath{"/tmp/edgellm.sock"};
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

// Read exactly n bytes or until delimiter
std::string readMessage(int clientSocket)
{
    std::string message;
    char buffer[4096];

    // First read the length prefix (4 bytes, big-endian)
    uint32_t msgLen = 0;
    ssize_t bytesRead = recv(clientSocket, &msgLen, sizeof(msgLen), MSG_WAITALL);
    if (bytesRead != sizeof(msgLen))
    {
        return "";
    }
    msgLen = ntohl(msgLen);

    if (msgLen > 10 * 1024 * 1024)
    { // 10MB limit
        return "";
    }

    // Read the message
    message.reserve(msgLen);
    size_t totalRead = 0;
    while (totalRead < msgLen)
    {
        ssize_t n = recv(clientSocket, buffer, std::min(sizeof(buffer), (size_t)(msgLen - totalRead)), 0);
        if (n <= 0)
            break;
        message.append(buffer, n);
        totalRead += n;
    }

    return message;
}

// Send message with length prefix
void sendMessage(int clientSocket, std::string const& message)
{
    uint32_t msgLen = htonl(message.length());
    send(clientSocket, &msgLen, sizeof(msgLen), 0);
    send(clientSocket, message.c_str(), message.length(), 0);
}

// Handle inference request
Json handleInference(Json const& request)
{
    Json response;
    response["error"] = nullptr;
    response["text"] = "";
    response["tokens"] = 0;

    try
    {
        // Extract parameters
        float temperature = request.value("temperature", 0.7f);
        float topP = request.value("top_p", 0.8f);
        int64_t topK = request.value("top_k", 50);
        int64_t maxTokens = request.value("max_tokens", 256);
        auto messages = request["messages"];

        // Convert to Edge-LLM format
        std::vector<rt::Message> llmMessages;
        std::vector<rt::imageUtils::ImageData> imageBuffers;

        for (auto const& msg : messages)
        {
            rt::Message llmMsg;
            llmMsg.role = msg["role"].get<std::string>();

            if (msg["content"].is_string())
            {
                rt::Message::MessageContent msgContent;
                msgContent.type = "text";
                msgContent.content = msg["content"].get<std::string>();
                llmMsg.contents.push_back(msgContent);
            }
            else if (msg["content"].is_array())
            {
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

                        // Load image if local path
                        if (!imagePath.empty() && imagePath.find("http") != 0 && imagePath.find("data:") != 0)
                        {
                            auto image = rt::imageUtils::loadImageFromFile(imagePath);
                            if (image.buffer != nullptr)
                            {
                                imageBuffers.push_back(std::move(image));
                            }
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

        // Check for thinking mode
        if (request.contains("enable_thinking"))
        {
            genRequest.enableThinking = request["enable_thinking"].get<bool>();
        }

        // Run inference with mutex protection
        rt::LLMGenerationResponse genResponse;
        {
            std::lock_guard<std::mutex> lock(gRuntime.mutex);
            if (!gRuntime.runtime->handleRequest(genRequest, genResponse, gRuntime.stream))
            {
                response["error"] = "Inference failed";
                return response;
            }
        }

        // Get output
        std::string outputText = genResponse.outputTexts.empty() ? "" : genResponse.outputTexts[0];
        int outputTokens = genResponse.outputIds.empty() ? 0 : genResponse.outputIds[0].size();

        response["text"] = outputText;
        response["tokens"] = outputTokens;
    }
    catch (std::exception const& e)
    {
        response["error"] = std::string("Exception: ") + e.what();
    }

    return response;
}

// Handle client connection
void handleClient(int clientSocket)
{
    std::string requestStr = readMessage(clientSocket);
    if (requestStr.empty())
    {
        close(clientSocket);
        return;
    }

    if (gConfig.debug)
    {
        std::cout << "[DEBUG] Received: " << requestStr.substr(0, 200) << "..." << std::endl;
    }

    try
    {
        Json request = Json::parse(requestStr);
        std::string command = request.value("command", "inference");

        Json response;
        if (command == "health")
        {
            response["status"] = "healthy";
            response["engine_loaded"] = gRuntime.initialized;
        }
        else if (command == "inference")
        {
            response = handleInference(request);
        }
        else
        {
            response["error"] = "Unknown command: " + command;
        }

        sendMessage(clientSocket, response.dump());
    }
    catch (std::exception const& e)
    {
        Json error;
        error["error"] = std::string("Parse error: ") + e.what();
        sendMessage(clientSocket, error.dump());
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
    std::cerr << "  --socketPath <path>       Unix socket path (default: /tmp/edgellm.sock)" << std::endl;
    std::cerr << "  --debug                   Enable debug logging" << std::endl;
}

int main(int argc, char* argv[])
{
    // Parse arguments
    static struct option options[] = {{"help", no_argument, 0, 'h'}, {"engineDir", required_argument, 0, 'e'},
        {"multimodalEngineDir", required_argument, 0, 'm'}, {"socketPath", required_argument, 0, 's'},
        {"debug", no_argument, 0, 'd'}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "he:m:s:d", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'h': printUsage(argv[0]); return 0;
        case 'e': gConfig.engineDir = optarg; break;
        case 'm': gConfig.multimodalEngineDir = optarg; break;
        case 's': gConfig.socketPath = optarg; break;
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
    std::cout << "TensorRT Edge-LLM Inference Server (Unix Socket)" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Engine Dir: " << gConfig.engineDir << std::endl;
    std::cout << "Visual Engine Dir: " << (gConfig.multimodalEngineDir.empty() ? "None" : gConfig.multimodalEngineDir)
              << std::endl;
    std::cout << "Socket Path: " << gConfig.socketPath << std::endl;
    std::cout << "============================================================" << std::endl;

    // Initialize runtime
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

    // Remove existing socket file
    unlink(gConfig.socketPath.c_str());

    // Create Unix socket
    int serverSocket = socket(AF_UNIX, SOCK_STREAM, 0);
    if (serverSocket < 0)
    {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }

    // Bind to socket path
    struct sockaddr_un serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sun_family = AF_UNIX;
    strncpy(serverAddr.sun_path, gConfig.socketPath.c_str(), sizeof(serverAddr.sun_path) - 1);

    if (bind(serverSocket, (struct sockaddr*) &serverAddr, sizeof(serverAddr)) < 0)
    {
        std::cerr << "Failed to bind to socket: " << gConfig.socketPath << std::endl;
        close(serverSocket);
        return 1;
    }

    // Set socket permissions (allow all users)
    chmod(gConfig.socketPath.c_str(), 0777);

    // Listen
    if (listen(serverSocket, 10) < 0)
    {
        std::cerr << "Failed to listen" << std::endl;
        close(serverSocket);
        return 1;
    }

    std::cout << "Server listening on: " << gConfig.socketPath << std::endl;
    std::cout << "Press Ctrl+C to stop" << std::endl;
    std::cout << "============================================================" << std::endl;

    // Accept connections
    while (gRunning)
    {
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
            int clientSocket = accept(serverSocket, nullptr, nullptr);
            if (clientSocket >= 0)
            {
                // Handle in separate thread
                std::thread(handleClient, clientSocket).detach();
            }
        }
    }

    // Cleanup
    close(serverSocket);
    unlink(gConfig.socketPath.c_str());
    CUDA_CHECK(cudaStreamDestroy(gRuntime.stream));
    std::cout << "Server stopped." << std::endl;

    return 0;
}
