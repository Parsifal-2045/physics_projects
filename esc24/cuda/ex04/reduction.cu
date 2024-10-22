// C++ standard headers
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <cassert>

// CUDA headers
#include <cuda_runtime.h>

// local headers
#include "cuda_check.h"

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

constexpr int numThreadsPerBlock = 1024;

// Part 4 of 8: implement the kernel
__global__ void block_sum(const int *input,
                          int *per_block_results,
                          const size_t n)
{
  // array shared among threads in a block
  __shared__ int sdata[numThreadsPerBlock];

  // each thread in the block sets an entry in the shared array to 0
  auto tid = threadIdx.x;
  sdata[tid] = 0;
  auto global_index = threadIdx.x + blockIdx.x * blockDim.x;

  // copy block-sized chunk of the input array in shared memory
  if (global_index < n)
  {
    sdata[tid] = input[global_index];
  }

  // wait for all threads to finish
  __syncthreads();

  // strided sum of all the elements of sdata
  for (unsigned int s = 1; s < blockDim.x; s *= 2)
  {
    int index = 2 * s * tid;
    if (index < blockDim.x)
    {
      sdata[index] += sdata[index + s];
    }

    /* % is expensive and half the threads are idle
     if (tid % (2 * s) == 0)
    {
      sdata[tid] += sdata[tid + s];
    }
    */

    // wait for all threads to finish before updating s
    __syncthreads();
  }

  // one thread per block writes the results in the shared array
  if (tid == 0)
  {
    per_block_results[blockIdx.x] = sdata[0];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(void)
{
  std::random_device rd;  // Will be used to obtain a seed for the random engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(-10, 10);
  // Create array of 256ki elements
  const long int num_elements = 1 << 18;
  const size_t memSize = num_elements * sizeof(int);
  // Generate random input on the host
  std::vector<int> h_input(num_elements);
  for (auto &elt : h_input)
  {
    elt = distrib(gen);
  }

  const int host_result = std::accumulate(h_input.begin(), h_input.end(), 0);
  std::cerr << "Host sum: " << host_result << std::endl;

  // Part 1 of 8: choose a device and create a CUDA stream
  CUDA_CHECK(cudaSetDevice(MYDEVICE));

  // Create a CUDA stream to execute asynchronous operations on this device
  cudaStream_t queue;
  CUDA_CHECK(cudaStreamCreate(&queue));

  // Part 2 of 8: copy the input data to device memory
  int *d_input;
  CUDA_CHECK(cudaMallocAsync(&d_input, memSize, queue));
  CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(), memSize, cudaMemcpyHostToDevice, queue));

  // Part 3 of 8: allocate memory for the partial sums
  const int numBlocks = (num_elements + numThreadsPerBlock - 1) / numThreadsPerBlock;

  // numOutputElements == numBlocks
  int numOutputElements = numBlocks;

  int *d_partial_sums_and_total;
  CUDA_CHECK(cudaMallocAsync(&d_partial_sums_and_total, numOutputElements * sizeof(int), queue));

  // Part 5 of 8: launch one kernel to compute, per-block, a partial sum.
  // How much shared memory does it need?
  block_sum<<<numBlocks, numThreadsPerBlock, 0, queue>>>(d_input, d_partial_sums_and_total,
                                                         num_elements);
  CUDA_CHECK(cudaGetLastError());

  // Part 6 of 8: compute the sum of the partial sums
  block_sum<<<1, numThreadsPerBlock, 0, queue>>>(d_partial_sums_and_total, d_partial_sums_and_total, numOutputElements);
  CUDA_CHECK(cudaGetLastError());

  // Part 7 of 8: copy the result back to the host
  int device_result = 0;
  CUDA_CHECK(cudaMemcpyAsync(&device_result, d_partial_sums_and_total, sizeof(int), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));
  std::cout << "Device sum: " << device_result << std::endl;
  assert(device_result == host_result);

  // Part 8 of 8: deallocate device memory and destroy the CUDA stream
  CUDA_CHECK(cudaFreeAsync(d_input, queue));
  CUDA_CHECK(cudaFreeAsync(d_partial_sums_and_total, queue));
  cudaStreamDestroy(queue);

  return 0;
}
