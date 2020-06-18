#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <list>
#include <cassert>

std::ostream &operator<<(std::ostream &os, std::vector<int> const &v)
{
    os << "{ ";
    for (auto const &a : v)
    {
        os << a << " ";
    }
    os << '}';
    return os;
}

std::ostream &operator<<(std::ostream &os, std::list<int> const &l)
{
    os << "[ ";
    for (auto const &a : l)
    {
        os << a << " ";
    }
    os << ']';
    return os;
}

int main()
{
    std::default_random_engine eng{};
    std::vector<int> v;
    std::generate_n(std::back_inserter(v), 8, eng);
    std::list<int> l;
    std::reverse_copy(v.begin(), v.end(), std::back_inserter(l));
    std::transform(v.begin(), v.end(), v.begin(), [](int n) { return n + 1; });
    std::cout << v << '\n';
    v.push_back(984943659);
    std::cout << v << '\n';
    auto it = std::adjacent_find(v.begin(), v.end(), [](int a, int b) { return b < a; });
    if (it != v.end())
    {
        std::cout << *it << '\n';
    }
    else
    {
        std::cout << "not found\n";
    }
    std::sort(v.begin(), v.end());
    std::cout << v << '\n';
    auto it2 = std::adjacent_find(v.begin(), v.end());
    if (it2 != v.end())
    {
        std::cout << *it2 << " at position " << std::distance(v.begin(), it2) << '\n';
    }
    else
    {
        std::cout << "not found\n";
    }
    //auto it3 = std::remove(v.begin(), v.end(), 984943659);
    //std::cout << std::distance(v.begin(), it3) << '\n';
    auto itp = std::partition(v.begin(), v.end(), [](int n) { return n % 2 == 0; });
    std::cout << v << '\n';
    std::cout << "Partition point " << std::distance(v.begin(), itp) << '\n';
    v.erase(std::remove_if(v.begin(), v.end(), [](int n) { return n % 2 == 0; }), v.end());
    std::cout << v << '\n';
    assert(std::none_of(v.begin(), v.end(), [](int n) { return n % 2 == 0; }));
    auto itp2 = std::partition(v.begin(), v.end(), [](int n) { return n % 2 == 1; });
    std::cout << "Partition point " << std::distance(v.begin(), itp2) << '\n';
}