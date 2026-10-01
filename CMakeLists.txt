cmake_minimum_required(VERSION 4.2)
project(GPT_2_From_Scratch)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)

add_executable(GPT_2_From_Scratch main.cpp
        Transformer/include/Tokenizer.h
        Transformer/src/Tokenizer.cpp
        Transformer/include/Codec.h
        Transformer/src/Codec.cpp
        Transformer/include/FileIO.h
        Transformer/src/FileIO.cpp
        NumCPP/include/Tensor.h
        NumCPP/src/Tensor.cpp
)

target_link_libraries(GPT_2_From_Scratch PRIVATE OpenMP::OpenMP_CXX)