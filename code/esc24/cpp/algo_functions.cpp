#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cassert>

std::ostream &operator<<(std::ostream &os, std::vector<int> const &c);
std::vector<int> make_vector(int N);

int main()
{
  // create a vector of N elements, generated randomly
  int const N = 10;
  std::vector<int> v = make_vector(N);
  std::cout << v << '\n';

  // multiply all the elements of the vector
  // use std::accumulate
  auto prod = std::accumulate(v.begin(), v.end(), int64_t{1}, [&](const auto a, const auto b)
                              { return a * b; });
  std::cout << "All elements multiplied: " << prod << '\n';
  // compute the mean and the standard deviation
  // use std::accumulate and a struct with two numbers to accumulate both the sum and the sum of squares
  std::pair<double, double> sum_sum2 =
      std::accumulate(v.begin(), v.end(), std::make_pair(double{}, double{}), [&](auto pair, auto value)
                      {
    auto sum = pair.first + value;
    auto sum2 = pair.second + value*value;
    return std::make_pair(sum, sum2); });
  double mean = sum_sum2.first / v.size();
  double std = std::sqrt(sum_sum2.second / v.size() - mean * mean);
  std::cout << "Mean: " << mean << ", std: " << std << '\n';

  // sort the vector in descending order
  // use std::sort
  std::sort(v.begin(), v.end(), [](int a, int b)
            { return b < a; });
  std::cout << "Vector sorted in descending order: " << v << '\n';

  // move the even numbers at the beginning of the vector
  // use std::partition
  auto pIt = std::partition(v.begin(), v.end(), [](int a)
                            { return a % 2 == 0; });
  std::cout << "Even numbers at the beginning (order not preserved): " << v << '\n';
  std::sort(v.begin(), pIt, [](int a, int b)
            { return b < a; });
  std::sort(pIt, v.end(), [](int a, int b)
            { return b < a; });
  std::cout << "Even numbers at the beginning, descending order sorting: " << v << '\n';

  // create another vector with the squares of the numbers in the first vector
  // use std::transform
  std::vector<int> another_v;
  another_v.reserve(v.size());
  std::transform(v.begin(), v.end(), std::back_inserter(another_v), [](int a)
                 { return a * a; });
  std::cout << "Vector of squares: " << another_v << '\n';

  // find the first multiple of 3 or 7
  // use std::find_if
  auto multiple = std::find_if(another_v.begin(), another_v.end(), [](int a)
                               { return a % 3 == 0 or a % 7 == 0; });
  assert(*multiple % 3 == 0 or *multiple % 7 == 0);
  std::cout << "Found multiple of 3 or 7 at position "
            << std::distance(another_v.begin(), multiple)
            << ", with value "
            << *multiple << '\n';

  // erase from the vector all the multiples of 3 or 7
  // use std::remove_if followed by vector::erase
  /*
  another_v.erase(std::remove_if(another_v.begin(), another_v.end(), [](int a)
                                 { return a % 3 == 0 or a % 7 == 0; }),
                  another_v.end());
  */

  //   or the newer std::erase_if utility (C++20)
  std::erase_if(another_v, [](int a)
                { return a % 3 == 0 or a % 7 == 0; });

  for (auto value : another_v)
  {
    assert(!(value % 3 == 0 or value % 7 == 0));
  }
  std::cout << "Removed all multiples of 3 or 7: " << another_v << '\n';
}

std::ostream &operator<<(std::ostream &os, std::vector<int> const &c)
{
  os << "{ ";
  std::copy(
      std::begin(c),
      std::end(c),
      std::ostream_iterator<int>{os, " "});
  os << '}';

  return os;
}

std::vector<int> make_vector(int N)
{
  // define a pseudo-random number generator engine and seed it using an actual
  // random device
  std::random_device rd;
  std::default_random_engine eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> dist{1, MAX_N};

  std::vector<int> result;
  result.reserve(N);
  std::generate_n(std::back_inserter(result), N, [&]
                  { return dist(eng); });

  return result;
}
