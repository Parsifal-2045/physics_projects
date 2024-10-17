#include <iostream>
#include <array>
#include <cassert>

constexpr int N = 10;

template <int N>
constexpr auto make_pascal()
{
    std::array<std::array<int, N + 1>, N + 1> result{};
    // top of the pyramid
    result[0][0] = 1;
    for (int i = 1; i != N + 1; ++i)
    {
        result[i][0] = 1;
        for (int j = 1; j != i; ++j)
        {
            result[i][j] = result[i - 1][j - 1] + result[i - 1][j];
        }
        result[i][i] = 1;
    }

    return result;
}

constexpr auto pascal_table = make_pascal<N>();

constexpr auto pascal(int i, int j)
{
    assert(i <= N + 1);
    assert(j <= N + 1);
    return pascal_table[i - 1][j - 1];
}

int main()
{
    for (int i = 0; i != N; ++i)
    {
        for (int j = 0; j != N; ++j)
        {
            if (pascal_table[i][j] != 0)
            {
                std::cout << pascal_table[i][j] << ' ';
            }
        }
        std::cout << '\n';
    }

    std::cout << pascal(4, 3) << '\n';
}