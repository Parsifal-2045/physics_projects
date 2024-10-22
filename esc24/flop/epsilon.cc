#include <iostream>
#include <utility>

std::pair<float, int> epsilon()
{
    float epsilon = 5.0f;

    int iters = 0;

    while (1.0f + epsilon > 1.0f)
    {
        epsilon /= 2.0f;
        ++iters;
    }
    return {epsilon, iters};
}

int main()
{
    std::cout << "The machine epsilon is 2 to the power of -"
              << epsilon().second
              << " or "
              << epsilon().first << '\n';
}