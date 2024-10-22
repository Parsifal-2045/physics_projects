#include <algorithm>
#include <chrono>
#include <cstdint>
//#include <format>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>
#include <execution>

template <typename T>
void axpyImpl(T a, T x, T y, T& z) {
  z = a * x + y;
}

template <typename T>
void axpy(auto policy, T a, std::vector<T> const& x, std::vector<T> const& y, std::vector<T>& z) {
  std::transform(policy, x.begin(), x.end(), y.begin(), z.begin(), [a](T x, T y) -> T {
    T z;
    axpyImpl(a, x, y, z);
    return z;
  });
}

template <typename T>
void measure(auto policy, T a, std::vector<T> const& x, std::vector<T> const& y) {
  std::vector<T> z(x.size(), 0);
  auto start = std::chrono::steady_clock::now();
  axpy(policy, a, x, y, z);
  auto finish = std::chrono::steady_clock::now();
  float ms = std::chrono::duration_cast<std::chrono::duration<float>>(finish - start).count() * 1000.f;
  //std::cout << std::format("{:6.1f}", ms) << " ms\n";
  std::cout << std::fixed << std::setprecision(1) << std::setw(6) << ms << " ms\n";
}

int main() {
  const std::size_t size = 100'000'000;
  const std::size_t times = 10;

  std::mt19937 gen{std::random_device{}()};
  std::uniform_real_distribution<float> dis{-std::numbers::pi, std::numbers::pi};
  float a = dis(gen);
  std::vector<float> x(size);
  std::ranges::generate(x, [&] { return dis(gen); });
  std::vector<float> y(size);
  std::ranges::generate(y, [&] { return dis(gen); });

  std::cout << "parallel saxpy\n";
  for (size_t i = 0; i < times; ++i)
    measure(std::execution::par, a, x, y);
  std::cout << '\n';

  std::cout << "sequential saxpy\n";
  for (size_t i = 0; i < times; ++i)
    measure(std::execution::seq, a, x, y);
  std::cout << '\n';

  std::cout << "unseq saxpy\n";
  for (size_t i = 0; i < times; ++i)
    measure(std::execution::unseq, a, x, y);
  std::cout << '\n';

  std::cout << "par_unseq saxpy\n";
  for (size_t i = 0; i < times; ++i)
    measure(std::execution::par_unseq, a, x, y);
  std::cout << '\n';
  // TODO
  //   - change the sequential loop to use std::execution::seq
  //   - try the other execution policies (par, unseq, par_unseq)
}
