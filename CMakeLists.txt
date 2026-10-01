cmake_minimum_required(VERSION 4.2)
project(GPT_2_From_Scratch)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)

add_executable(GPT_2_From_Scratch main.cpp
        Tokenizer/include/Tokenizer.h
        Tokenizer/src/Tokenizer.cpp
        Tokenizer/include/Codec.h
        Tokenizer/src/Codec.cpp
        Tokenizer/include/FileIO.h
        Tokenizer/src/FileIO.cpp
        NumCPP/include/Tensor.h
        NumCPP/src/Tensor.cpp
        NeuralNet/include/Layer.h
        NeuralNet/include/Linear.h
        NeuralNet/include/Embedding.h
        NeuralNet/include/LayerNorm.h
        NeuralNet/include/Dropout.h
        NeuralNet/include/Sequential.h
)

target_link_libraries(GPT_2_From_Scratch PRIVATE OpenMP::OpenMP_CXX)