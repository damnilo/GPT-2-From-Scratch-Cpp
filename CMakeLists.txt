cmake_minimum_required(VERSION 3.24)
project(GPT_2_From_Scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    message(STATUS "CUDA GEMM enabled with ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "CUDA compiler not found; matrix multiplication stays on the CPU")
endif()

set(GPT_SOURCES
        Tokenizer/include/Tokenizer.h
        Tokenizer/src/Tokenizer.cpp
        Tokenizer/include/Codec.h
        Tokenizer/src/Codec.cpp
        Tokenizer/include/FileIO.h
        Tokenizer/src/FileIO.cpp
        NumCPP/include/Tensor.h
        NumCPP/src/Tensor.cpp
        NeuralNet/Layers/include/Layer.h
        NeuralNet/Layers/include/Linear.h
        NeuralNet/Layers/include/Embedding.h
        NeuralNet/Layers/include/LayerNorm.h
        NeuralNet/Layers/include/Dropout.h
        NeuralNet/Layers/include/Sequential.h
        NeuralNet/Layers/src/Embedding.cpp
        NeuralNet/Layers/src/Linear.cpp
        NeuralNet/Layers/src/LayerNorm.cpp
        NeuralNet/Layers/src/Sequential.cpp
        NeuralNet/Layers/include/ReLU.h
        NeuralNet/Layers/src/ReLU.cpp
        NeuralNet/Layers/src/GeLU.cpp
        NeuralNet/Layers/include/GeLU.h
        NeuralNet/Layers/src/Dropout.cpp
        NeuralNet/Layers/include/Softmax.h
        NeuralNet/Layers/src/Softmax.cpp
        NumCPP/include/Math.h
        NumCPP/src/Math.cpp
        NeuralNet/Losses/include/CrossEntropyLoss.h
        NeuralNet/Optimizers/include/AdamW.h
        NeuralNet/Losses/src/CrossEntropyLoss.cpp
        NeuralNet/Optimizers/src/AdamW.cpp
        Transformer/include/Attention.h
        Transformer/src/Attention.cpp
        Transformer/include/MultiHeadAttention.h
        Transformer/src/MultiHeadAttention.cpp
        Transformer/include/Transformer.h
        Transformer/src/Transformer.cpp
        Transformer/include/FeedForward.h
        Transformer/src/FeedForward.cpp
        Transformer/include/PositionalEmbedding.h
        Transformer/src/PositionalEmbedding.cpp
        Transformer/include/Model.h
        Transformer/src/Model.cpp
        Transformer/include/Checkpoint.h
        Transformer/src/Checkpoint.cpp
)

add_executable(GPT_2_From_Scratch main.cpp train.cpp ${GPT_SOURCES})
add_executable(bench bench/bench.cpp ${GPT_SOURCES})
target_include_directories(bench PRIVATE ${CMAKE_SOURCE_DIR})
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(bench PRIVATE -finput-charset=UTF-8 -fexec-charset=UTF-8)
endif()

add_executable(GPT_2_Tests Tests/tests.cpp
        NumCPP/src/Tensor.cpp
        NeuralNet/Layers/src/Embedding.cpp
        NeuralNet/Layers/src/Linear.cpp
        NeuralNet/Layers/src/LayerNorm.cpp
        NeuralNet/Layers/src/GeLU.cpp
        NeuralNet/Layers/src/Softmax.cpp
)

file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}/Tests/references/data" TEST_DATA_DIR)
target_compile_definitions(GPT_2_Tests PRIVATE "DATA_DIR=\"${TEST_DATA_DIR}\"")

if(CMAKE_CUDA_COMPILER)
    target_sources(GPT_2_From_Scratch PRIVATE NumCPP/src/CudaGemm.cu)
    target_sources(GPT_2_Tests PRIVATE NumCPP/src/CudaGemm.cu)
    target_sources(bench PRIVATE NumCPP/src/CudaGemm.cu)
    target_compile_definitions(bench PRIVATE GPT_HAS_CUDA=1)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(GPT_2_From_Scratch PRIVATE OpenMP::OpenMP_CXX CUDA::cublas)
    target_link_libraries(GPT_2_Tests PRIVATE OpenMP::OpenMP_CXX CUDA::cublas)
    target_link_libraries(bench PRIVATE OpenMP::OpenMP_CXX CUDA::cublas)
else()
    target_sources(GPT_2_From_Scratch PRIVATE NumCPP/src/CudaGemmStub.cpp)
    target_sources(GPT_2_Tests PRIVATE NumCPP/src/CudaGemmStub.cpp)
    target_sources(bench PRIVATE NumCPP/src/CudaGemmStub.cpp)
    target_compile_definitions(bench PRIVATE GPT_HAS_CUDA=0)
    target_link_libraries(GPT_2_From_Scratch PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(GPT_2_Tests PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(bench PRIVATE OpenMP::OpenMP_CXX)
endif()