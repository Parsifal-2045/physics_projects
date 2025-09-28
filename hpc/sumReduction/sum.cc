#include <chrono>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <type_traits>
#include <vector>

template <typename T> T seq_sum(const T *v, long int n) {
  T sum = 0;
  for (int i = 0; i != n; i++) {
    sum += v[i];
  }
  return sum;
}

// Parallel sum using OpenMP (wrong implementation, no atomic operation)
template <typename T> T par_sum_concurrent(const T *v, long int n) {
  T result = 0;
#pragma omp parallel
  {
    const int block_size = n / omp_get_num_threads();
    const int start = omp_get_thread_num() * block_size;
    const int end = start + block_size;
    for (int i = start; i < end; i++) {
      result += v[i];
    }
  }
  return result;
}

// Parallel sum using OpenMP (wrong implementation, but with mutex)
template <typename T> T par_sum_concurrent_mutex(const T *v, long int n) {
  T result = 0;
#pragma omp parallel
  {
    const int block_size = n / omp_get_num_threads();
    const int start = omp_get_thread_num() * block_size;
    const int end = start + block_size;
    for (int i = start; i < end; i++) {
#pragma omp atomic
      result += v[i];
    }
  }
  return result;
}

// Parallel sum using OpenMP (correct work division, with mutex)
template <typename T> T par_sum_ceil_mutex(const T *v, long int n) {
  T result = 0;
#pragma omp parallel
  {
    const int tIdx = omp_get_thread_num();
    const int nThreads = omp_get_num_threads();

    const int start = n * tIdx / nThreads;
    const int end = n * (tIdx + 1) / nThreads;
    for (int i = start; i < end; i++) {
#pragma omp atomic
      result += v[i];
    }
  }
  return result;
}

// Parallel sum using OpenMP (correct work division, with global mutex)
template <typename T> T par_sum_global_mutex(const T *v, long int n) {
  T result = 0;
#pragma omp parallel
  {
    const int tIdx = omp_get_thread_num();
    const int nThreads = omp_get_num_threads();

    const int start = n * tIdx / nThreads;
    const int end = n * (tIdx + 1) / nThreads;
    T partial_sum = 0;
    for (int i = start; i < end; i++) {
      partial_sum += v[i];
    }
#pragma omp atomic
    result += partial_sum;
  }
  return result;
}

// Parallel sum using OpenMP (wrong implementation, t0 does the reduction to
// avoid mutex with no barrier)
template <typename T> T par_sum_reduction_t0(const T *v, long int n) {
  const int nThreads = omp_get_num_threads();
  const int tIdx = omp_get_thread_num();
  T partial_sums[nThreads];
  T result = 0;

#pragma omp parallel
  {
    const int start = n * tIdx / nThreads;
    const int end = n * (tIdx + 1) / nThreads;
    T partial_sum = 0;
    for (int i = start; i < end; i++) {
      partial_sum += v[i];
    }
    partial_sums[tIdx] = partial_sum;
    // no barrier here, so t0 might start before others have written
    if (tIdx == 0) {
      for (int i = 0; i < nThreads; i++)
        result += partial_sums[i];
    }
  }
  return result;
}

// Parallel sum using OpenMP (correct reduction with explicit barrier)
template <typename T>
T par_sum_reduction_explicit_barrier(const T *v, long int n) {
  const int nThreads = omp_get_num_threads();
  const int tIdx = omp_get_thread_num();
  T partial_sums[nThreads];
  T result = 0;

#pragma omp parallel
  {
    const int start = n * tIdx / nThreads;
    const int end = n * (tIdx + 1) / nThreads;
    T partial_sum = 0;
    for (int i = start; i < end; i++) {
      partial_sum += v[i];
    }
    partial_sums[tIdx] = partial_sum;
#pragma omp barrier // added barrier
    if (tIdx == 0) {
      for (int i = 0; i < nThreads; i++)
        result += partial_sums[i];
    }
  }
  return result;
}

// Parallel sum using OpenMP (correct reduction with implicit barrier)
template <typename T> T par_sum_reduction_barrier(const T *v, long int n) {
  const int nThreads = omp_get_num_threads();
  const int tIdx = omp_get_thread_num();
  T partial_sums[nThreads];
  T result = 0;

#pragma omp parallel
  {
    const int start = n * tIdx / nThreads;
    const int end = n * (tIdx + 1) / nThreads;
    T partial_sum = 0;
    for (int i = start; i < end; i++) {
      partial_sum += v[i];
    }
    partial_sums[tIdx] = partial_sum;
  } // implicit barrier at the end of the parallel region
  // only t0 will do the reduction
  for (int i = 0; i < nThreads; i++)
    result += partial_sums[i];

  return result;
}

// Parallel sum using OpenMP (correct reduction with OpenMP reduction clause)
template <typename T> T par_sum_reduction(const T *v, long int n) {
  T result = 0;
#pragma omp parallel for reduction(+ : result)
  for (int i = 0; i < n; i++) {
    result += v[i];
  }
  return result;
}

int main() {
  const long int N = 939391;
  std::vector<std::remove_const_t<decltype(N)>> v(N);
  std::iota(v.begin(), v.end(),
            static_cast<std::remove_const_t<decltype(N)>>(0));

  // Sequential sum
  auto start = std::chrono::high_resolution_clock::now();
  float result = seq_sum(v.data(), N);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "------------- Sequential sum -------------\n"
            << " Elapsed time: " << elapsed.count() << " seconds\n"
            << "Result: " << result << '\n';
  // Parallel sum with concurrent updates
  start = std::chrono::high_resolution_clock::now();
  result = par_sum_concurrent(v.data(), N);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "------------- [WRONG] Parallel sum with concurrent updates "
               "-------------\n"
            << " Elapsed time: " << elapsed.count() << " seconds\n"
            << "Result: " << result << '\n';
  // Parallel sum with mutex
  start = std::chrono::high_resolution_clock::now();
  result = par_sum_concurrent_mutex(v.data(), N);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout
      << "------------- [WRONG] Parallel sum with mutex but wrong partitioning "
         "-------------\n"
      << " Elapsed time: " << elapsed.count() << " seconds\n"
      << "Result: " << result << '\n';
  // Parallel sum with ceil and mutex
  start = std::chrono::high_resolution_clock::now();
  result = par_sum_ceil_mutex(v.data(), N);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "------------- Parallel sum with ceil and mutex "
               "-------------\n"
            << " Elapsed time: " << elapsed.count() << " seconds\n"
            << "Result: " << result << '\n';
  // Parallel sum with global mutex
  start = std::chrono::high_resolution_clock::now();
  result = par_sum_global_mutex(v.data(), N);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "------------- Parallel sum with global mutex "
               "-------------\n"
            << " Elapsed time: " << elapsed.count() << " seconds\n"
            << "Result: " << result << '\n';
  // Parallel sum wrong with no barrier with t0 doing reduction to avoid mutex
  start = std::chrono::high_resolution_clock::now();
  result = par_sum_reduction_t0(v.data(), N);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "------------- [WRONG] Parallel sum with reduction done by t0 "
               "to avoid mutex "
               "-------------\n"
            << " Elapsed time: " << elapsed.count() << " seconds\n"
            << "Result: " << result << '\n';
  // Parallel sum with reduction and barrier
  start = std::chrono::high_resolution_clock::now();
  result = par_sum_reduction_barrier(v.data(), N);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout
      << "------------- Parallel sum with reduction done by t0 and barrier "
         "-------------\n"
      << " Elapsed time: " << elapsed.count() << " seconds\n"
      << "Result: " << result << '\n';
  // Parallel sum reduction with OpenMP reduction clause
  start = std::chrono::high_resolution_clock::now();
  result = par_sum_reduction(v.data(), N);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "------------- Parallel sum with openMP reduction"
               "-------------\n"
            << " Elapsed time: " << elapsed.count() << " seconds\n"
            << "Result: " << result << '\n';
}
