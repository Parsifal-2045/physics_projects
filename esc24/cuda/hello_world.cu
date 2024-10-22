#include <iostream>

__global__ void mykernel()
{
    printf("Hello world!\n");
}

int main()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    mykernel<<<1, 1, 0>>>();
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}