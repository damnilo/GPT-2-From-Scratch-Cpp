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
        NeuralNet/src/Embedding.cpp
        NeuralNet/src/Linear.cpp
        NeuralNet/src/LayerNorm.cpp
        NeuralNet/src/Sequential.cpp
        NeuralNet/include/ReLU.h
        NeuralNet/src/ReLU.cpp
        NeuralNet/src/GeLU.cpp
        NeuralNet/include/GeLU.h
        NeuralNet/src/Dropout.cpp
        NeuralNet/include/Softmax.h
        NeuralNet/src/Softmax.cpp
        NumCPP/include/Math.h
        NumCPP/src/Math.cpp
)

target_link_libraries(GPT_2_From_Scratch PRIVATE OpenMP::OpenMP_CXX)