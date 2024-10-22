#include <iostream>

constexpr int nSteps = 100'000'000;

constexpr float float_forward_sum(const int nSteps)
{
    float sum{};
    for (int i = 1; i != nSteps; ++i)
    {
        sum += 1.0f / i;
    }
    return sum;
}

constexpr float float_backward_sum(const int nSteps)
{
    float sum{};
    for (int i = nSteps; i != 0; --i)
    {
        sum += 1.0f / i;
    }
    return sum;
}

constexpr double double_forward_sum(const int nSteps)
{
    double sum{};
    for (int i = 1; i != nSteps; ++i)
    {
        sum += 1.0f / i;
    }
    return sum;
}

constexpr double double_backward_sum(const int nSteps)
{
    double sum{};
    for (int i = nSteps; i != 0; --i)
    {
        sum += 1.0f / i;
    }
    return sum;
}

int main()
{
    std::cout << "Sum forward (float): " << float_forward_sum(nSteps) << '\n';
    std::cout << "Sum backward (float) : " << float_backward_sum(nSteps) << '\n';

    std::cout << "Sum forward (double): " << double_forward_sum(nSteps) << '\n';
    std::cout << "Sum backward (double) : " << double_backward_sum(nSteps) << '\n';
}