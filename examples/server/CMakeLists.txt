# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

# HTTP Server (standalone)
add_executable(llm_server llm_server.cpp)

target_include_directories(llm_server PRIVATE ${COMMON_INCLUDE_DIRS}
                                              ${CMAKE_SOURCE_DIR}/examples)
target_link_libraries(llm_server PRIVATE edgellmCore edgellmTokenizer
                                         ${NVINFER_LIB} ${NV_ONNX_PARSER_LIB}
                                         ${CUDART_LIB} ${CUDA_DRIVER_LIB}
                                         uuid)
add_cross_build_link_options(llm_server)

# Unix Socket Server (for Python API integration)
add_executable(llm_inference_server llm_inference_server.cpp)

target_include_directories(llm_inference_server PRIVATE ${COMMON_INCLUDE_DIRS}
                                                        ${CMAKE_SOURCE_DIR}/examples)
target_link_libraries(llm_inference_server PRIVATE edgellmCore edgellmTokenizer
                                                   ${NVINFER_LIB} ${NV_ONNX_PARSER_LIB}
                                                   ${CUDART_LIB} ${CUDA_DRIVER_LIB})
add_cross_build_link_options(llm_inference_server)
