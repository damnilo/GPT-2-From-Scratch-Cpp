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
)

target_link_libraries(GPT_2_From_Scratch PRIVATE OpenMP::OpenMP_CXX)