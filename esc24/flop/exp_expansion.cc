#include <iostream>
#include <math.h>

constexpr float myExp(const float x)
{
    long iterations = 0;
    float delta = 1.f;
    float exp = 1.f;
    while ((1.0f + delta) != 1.0f)
    {
        ++iterations;
        delta *= x / iterations;
        exp += delta;
    }
    return exp;
}

int main()
{
    std::cout << std::exp(5) << ", " << myExp(5) << '\n';
    std::cout << std::exp(10) << ", " << myExp(10) << '\n';
    std::cout << std::exp(15) << ", " << myExp(15) << '\n';
    std::cout << std::exp(20) << ", " << myExp(20) << '\n';
    std::cout << std::exp(-5) << ", " << myExp(-5) << '\n';
    std::cout << std::exp(-10) << ", " << myExp(-10) << '\n'; // breaks down for x < 0
    std::cout << std::exp(-15) << ", " << myExp(-15) << '\n'; // breaks down for x < 0
    std::cout << std::exp(-20) << ", " << myExp(-20) << '\n'; // breaks down for x < 0
}