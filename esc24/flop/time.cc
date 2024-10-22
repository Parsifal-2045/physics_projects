#include <iostream>

constexpr int N = 1'000'000;

constexpr float tick = 1.0f/128.0f;

constexpr float time(int nTicks)
{
    float time{};
    for (int i = 0; i != nTicks; ++i)
    {
        time += tick;
    }
    return time;
}

int main()
{
    std::cout << time(N) << '\n';
}