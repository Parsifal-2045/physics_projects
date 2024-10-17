#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <execution>
#include <chrono>
#include <cassert>

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<float>;

int main()
{
  // define a pseudo-random number generator engine and seed it using an actual
  // random device
  std::random_device rd;
  std::default_random_engine eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> uniform_dist{1, MAX_N};

  // fill a vector with SIZE random numbers
  int const SIZE = 10'000'000;
  std::vector<int> v;
  v.reserve(SIZE);
  std::generate_n(std::back_inserter(v), SIZE, [&]
                  { return uniform_dist(eng); });

  {
    auto t0 = Clock::now();
    // sum all the elements of the vector with std::accumulate
    std::accumulate(v.begin(), v.end(), 0);
    auto t1 = Clock::now();
    Duration d = t1 - t0;
    std::cout << "Accumulate in " << d.count() << " s\n";
  }

  {
    auto t0 = Clock::now();
    // sum all the elements of the vector with std::reduce, sequential policy
    // NB you need to pass the initial value
    std::reduce(std::execution::seq, v.begin(), v.end(), 0);
    auto t1 = Clock::now();
    Duration d = t1 - t0;
    std::cout << "Sequential reduce in " << d.count() << " s\n";
  }

  {
    auto t0 = Clock::now();
    // sum all the elements of the vector with std::reduce, parallel policy
    // NB you need to pass the initial value
    std::reduce(std::execution::par, v.begin(), v.end(), 0);
    auto t1 = Clock::now();
    Duration d = t1 - t0;
    std::cout << "Parallel reduce in " << d.count() << " s\n";
  }

  {
    auto copy = v;
    auto t0 = Clock::now();
    // sort the vector with std::sort
    std::sort(v.begin(), v.end());
    auto t1 = Clock::now();
    Duration d = t1 - t0;
    std::cout << "Sort in " << d.count() << " s\n";
  }

  {
    auto copy = v;
    auto t0 = Clock::now();
    // sort the vector with std::sort, sequential policy
    std::sort(std::execution::seq, v.begin(), v.end());
    auto t1 = Clock::now();
    Duration d = t1 - t0;
    std::cout << "Sequential sort in " << d.count() << " s\n";
  }

  {
    auto copy = v;
    auto t0 = Clock::now();
    // sort the vector with std::sort, parallel policy
    std::sort(std::execution::par, v.begin(), v.end());
    auto t1 = Clock::now();
    Duration d = t1 - t0;
    std::cout << "Parallel sort in " << d.count() << " s\n";
  }
}
