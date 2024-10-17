#include <iostream>
#include <random>

class LinearCongruential
{
public:
  LinearCongruential(unsigned long seed = 1) : seed_(seed), x_(seed) {}

  unsigned long operator()()
  {
    return x_ = (16807UL * x_) % ((1UL << 31) - 1);
  }

  int getSeed()
  {
    return seed_;
  }

private:
  unsigned long seed_;
  unsigned long x_;
};

int main()
{
  LinearCongruential eng;
  std::default_random_engine stdeng;
  std::cout << eng() << '\t' << stdeng() << '\n';
  std::cout << eng() << '\t' << stdeng() << '\n';
  std::cout << eng() << '\t' << stdeng() << '\n';
  std::cout << eng() << '\t' << stdeng() << '\n';
}