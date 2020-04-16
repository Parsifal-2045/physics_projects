#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>

// Lambda expressions

/*
int main()
{
    std::vector<int> v(10);
    std::iota(v.begin(), v.end(), 0);
    std::vector<std::string> s(v.size());
    std::string name;
    std::cin >> name;
    std::transform(v.begin(), v.end(), s.begin(), [=](int n) { return name + std::to_string(n); });
    for (auto &str : s)
    {
        std::cout << str << '\n';
    }
}
*/