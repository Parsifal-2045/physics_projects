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

  // sum all the elements of the vector
  // use std::accumulate
  int sum = std::accumulate(v.begin(), v.end(), 0);
  std::cout << "Sum of all the elements: " << sum << '\n';

  // compute the average of the first half and of the second half of the vector
  const auto mid_it = v.begin() + v.size() / 2;
  const float first_avg = static_cast<float>(std::accumulate(v.begin(), mid_it, 0)) / std::distance(v.begin(), mid_it);
  const float second_avg = static_cast<float>(std::accumulate(mid_it, v.end(), 0)) / std::distance(v.begin(), mid_it);
  std::cout << "Average of the first half: " << first_avg << ", average of the second half:  " << second_avg << '\n';
  assert(first_avg * v.size() / 2 + second_avg * v.size() / 2 == sum);

  // move the three (four) central elements to the beginning of the vector
  // use std::rotate
  assert(v.size() >= 4);
  std::rotate(v.begin(), std::prev(mid_it, 2), std::next(mid_it, 2));
  std::cout << "Rotate four central elements to the beginning: " << v << '\n';

  // remove duplicate elements
  // use std::sort followed by std::unique/unique_copy
  std::sort(v.begin(), v.end());
  std::cout << "Sorted vector: " << v << '\n';

  // remove duplicates in place
  v.erase(std::unique(v.begin(), v.end()), v.end());
  std::cout << "Duplicates removed in place: " << v << '\n';

  // remove duplicates in copy
  std::vector<int> unique_v;
  unique_v.reserve(v.size());
  std::unique_copy(v.begin(), v.end(), std::back_inserter(unique_v));
  assert(unique_v.size() == v.size());
  std::cout << "Unique copy: " << unique_v << '\n';

  for (size_t i = 0; i != v.size(); ++i)
  {
    assert(v[i] == unique_v[i]);
  }
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
