cmake_minimum_required(VERSION 3.24)
project(cuda_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

function(add_lab name)    
    add_executable(${name} src/${name}.cu)
    target_include_directories(${name} PRIVATE include)
    target_link_libraries(${name} PRIVATE CUDA::cudart CUDA::cublas)
endfunction()

add_lab(bench)