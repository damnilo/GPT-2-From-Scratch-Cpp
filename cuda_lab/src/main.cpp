#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device Name: %s\n", prop.name);
    printf("Device Total Memory: %zu\n", prop.totalGlobalMem);
    printf("Device Shared Memory per Block: %zu\n", prop.sharedMemPerBlock);
    printf("Device Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Device Max Threads Dim: %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Device Max Grid Size: %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Device Clock Rate: %d\n", prop.clockRate);
    printf("Device Compute Mode: %s\n", prop.computeMode);
    printf("Device Total Constant Memory: %zu\n", prop.totalConstMem);
    printf("Device Texture Alignment: %zu\n", prop.textureAlignment);
    printf("Device Concurrent Kernels: %d\n", prop.concurrentKernels);
    printf("Device Concurrent Managed: %d\n", prop.concurrentManaged);
}