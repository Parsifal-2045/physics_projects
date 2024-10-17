#include <cstdlib>
#include <iostream>
#include <utility>

constexpr double pi(int n)
{
  auto const step = 1. / n;
  auto sum = 0.;
  for (int i = 0; i != n; ++i)
  {
    auto x = (i + 0.5) * step;
    sum += 4. / (1. + x * x);
  }

  return step * sum;
}

int main()
{
  static_assert(pi(10000) > 0);
  constexpr auto p = pi(10000);
  std::cout << p << '\n';
}
