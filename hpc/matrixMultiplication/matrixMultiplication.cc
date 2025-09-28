#include <chrono>
#include <iostream>
#include <random>
#include <vector>

// Measure cache performance with and without matrix transposition
// clang-format off
// perf stat -B -e task-clock,cycles,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses
// clang-format on

#ifdef TRANSPOSE
template <typename T>
void multiplyMatrices(const T *p, const T *q, T *r, int n) {
  T *qT = new T[n * n];
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      qT[j * n + i] = q[i * n + j];
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      r[i * n + j] = 0;
      for (int k = 0; k < n; ++k) {
        r[i * n + j] += p[i * n + k] * qT[j * n + k];
      }
    }
  }
  delete[] qT;
}
#else
template <typename T>
void multiplyMatrices(const T *p, const T *q, T *r, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      r[i * n + j] = 0;
      for (int k = 0; k < n; ++k) {
        r[i * n + j] += p[i * n + k] * q[k * n + j];
      }
    }
  }
}
#endif

int main() {
  int N = 1024;
  std::vector<double> a(N * N);
  std::vector<double> b(N * N);
  std::vector<double> c(N * N, 0);
  // Initialize matrices a and b with random values
  // fixed seed for reproducibility
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int i = 0; i < N * N; ++i) {
    a[i] = 100 * dist(rng);
    b[i] = 100 * dist(rng);
  }

  int repetitions = 10;
  std::vector<std::chrono::duration<double>> iterationTimes(repetitions);
  // skip first iteration for more accurate timing
  auto start = std::chrono::high_resolution_clock::now();
  multiplyMatrices(a.data(), b.data(), c.data(), N);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  for (int iter = 0; iter < repetitions; ++iter) {
    start = std::chrono::high_resolution_clock::now();
    multiplyMatrices(a.data(), b.data(), c.data(), N);
    end = std::chrono::high_resolution_clock::now();
    iterationTimes[iter] = end - start;
  }
  elapsed = std::accumulate(iterationTimes.begin(), iterationTimes.end(),
                            std::chrono::duration<double>(0));

#ifdef TRANSPOSE
  std::cout << "------------- Using matrix transposition -------------\n";
#else
  std::cout << "----------- Not using matrix transposition -----------\n";
#endif
  std::cout << "First element of result matrix: " << c[0] << "\n";
  std::cout << "Average elapsed time (running " << repetitions + 1
            << " measurements and skipping the "
               "first one): "
            << elapsed.count() / repetitions << " seconds\n";
}
