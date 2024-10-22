#include <cstdlib>
#include <iostream>
#include <span> // c++20
#include <memory>

void do_something_with(std::span<int> a);

int main()
{
  // allocate memory for 1000 int's
  int const SIZE = 1000;
  std::unique_ptr<int[]> p = std::make_unique<int[]>(SIZE);
  // auto p = static_cast<int *>(std::malloc(SIZE * sizeof(int)));
  do_something_with({p.get(), SIZE});
  // std::free(p);
}

void do_something_with(std::span<int> a)
{
  std::fill(a.begin(), a.end(), 42);
}