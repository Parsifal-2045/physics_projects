#include <iostream>
#include <cassert>

__global__ void add(const int *a, const int *b, int *c, const int n)
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
    {
        c[index] = a[index] + b[index];
    }
}

int main()
{
    long long int N = (1 << 28) + 1;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    size_t size = N * sizeof(int);

    // Allocate pinned memory on host
    cudaMallocHost(&a, size);
    cudaMallocHost(&b, size);
    cudaMallocHost(&c, size);
    for (int i = 0; i != N; ++i)
    {
        a[i] = i;
        b[i] = 8 * i;
    }

    // Allocate memory on device
    cudaMallocAsync(&d_a, size, stream);
    cudaMallocAsync(&d_b, size, stream);
    cudaMallocAsync(&d_c, size, stream);

    // Copy inputs to device
    cudaMemcpyAsync(d_a, a, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b, size, cudaMemcpyHostToDevice, stream);

    // Launch kernel on GPU
    // Tell the scheduler what resources each block needs
    // <<<nBlocks, nThreads per block, max amount of shared memory to use, CUDA stream>>>
    int nThreadsPerBlock = 512;
    int nBlocks = (N + nThreadsPerBlock - 1) / nThreadsPerBlock;
    size_t maxDinamicSharedMem = 0;
    add<<<nBlocks, nThreadsPerBlock, maxDinamicSharedMem, stream>>>(d_a, d_b, d_c, N);

    // Retrieve results
    cudaMemcpyAsync(c, d_c, size, cudaMemcpyDeviceToHost, stream);

    // Free device memory
    cudaFreeAsync(d_a, stream);
    cudaFreeAsync(d_b, stream);
    cudaFreeAsync(d_c, stream);

    // Synch to be able to use c on host
    cudaStreamSynchronize(stream);
    for (int i = 0; i != N; ++i)
    {
        assert(a[i] + b[i] == c[i]);
        // std::cout << a[i] << "+" << b[i] << "=" << c[i] << '\n';
    }
    std::cout << "Correct result!" << '\n';

    // Destroy the stream and free pinned host memory
    cudaStreamDestroy(stream);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
}