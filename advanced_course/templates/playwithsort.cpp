#include "OrderedVector.hpp"
#include "Complex.hpp"
#include <string>
#include <iostream>
#include <algorithm>

/*
struct OrderFunctor {
    bool operator() (const TYPE &s, const TYPE &t) const {
    }
};
*/

int main()
{
    std::cout << "Integer\n";
    OrderedVector<int> v(10);
    for (int i = 10; i > 0; i--)
        v.add(i);
    for (int i = 0; i < 10; i++)
        std::cout << v[i] << " ";
    std::cout << "\n\n";

    std::cout << "String\n";
    OrderedVector<std::string> vs(5);
    vs.add(std::string("one"));
    vs.add(std::string("two"));
    vs.add(std::string("three"));
    vs.add(std::string("four"));
    vs.add(std::string("five"));
    for (int i = 0; i < 5; i++)
        std::cout << vs[i] << " ";
    std::cout << "\n\n";

    std::cout << "Complex" << '\n';
    OrderedVector<Complex> c(10);
    for (int i = 10; i != 0; i--)
    {
        auto f = static_cast<float>(i);
        c.add(Complex{f / 2, 2 * f});
    }
    for (int i = 0; i != 10; i++)
    {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;
}
