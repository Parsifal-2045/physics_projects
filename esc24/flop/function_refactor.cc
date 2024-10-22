// implement f(x) = 1 / (sqrt(x^2 + 1) - x)

#include <math.h>
#include <cstdio>

constexpr long N1 = 10e5;
constexpr long N2 = 10e6;
constexpr long N3 = 10e7;
constexpr long N4 = 10e8;

constexpr float naiveFunc(long x)
{
    return 1.0 / (std::sqrt(x * x + 1) - x);
}

constexpr float refactoredFunc(long x)
{
    return std::sqrt(x * x + 1) + x;
}

int main()
{
    printf("N: %ld\n", N1);
    printf("Naive func: %f, Refactored func: %f\n", naiveFunc(N1), refactoredFunc(N1));

    printf("N: %ld\n", N2);
    printf("Naive func: %f, Refactored func: %f\n", naiveFunc(N2), refactoredFunc(N2));

    printf("N: %ld\n", N3);
    printf("Naive func: %f, Refactored func: %f\n", naiveFunc(N3), refactoredFunc(N3));

    printf("N: %ld\n", N4);
    printf("Naive func: %f, Refactored func: %f\n", naiveFunc(N4), refactoredFunc(N4));
}