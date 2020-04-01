#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <random>
#include <algorithm>
#include <cassert>

template <class Container>
std::string to_string(Container const &cont)
{

    // [v1 v2 v3 ... vn ]
    std::string result{"{ "};
    for (auto const &v : cont)
    {
        result += std::to_string(v);
        result += ' ';
    }
    result += '}';
    return result;
}

int main()
{
    std::array<int, 16> a;
    std::iota(a.begin(), a.end(), 0);
    std::cout << to_string(a) << '\n';
    auto b = a;
    std::cout << to_string(b) << '\n';
    auto const beg = std::begin(b);
    auto const end = std::end(b);
    std::mt19937 eng;
    std::shuffle(beg, end, eng);
    std::cout << to_string(b) << '\n';
    std::cout << std::boolalpha << std::is_sorted(beg,end) << '\n';
    auto it = std::is_sorted_until(beg, end);
    std::cout << std::distance(beg, it) << '\n';
    std::nth_element(beg, beg + 5, end);
    std::cout << to_string(b) << '\n';
    std::sort(beg, end);
    assert(a == b);
    {
    std::vector<int> v(16);
    std::copy(beg, end, std::begin(v));
    std:: cout << to_string(v) << '\n';
    }
}