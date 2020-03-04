#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
//#include <iostream>

auto sum(int const &a, int const &b)
{
    int r = a + b;
    return r;
}

/*int main()
{
    int a, b;
    std::cout << "Inserire il primo termine: ";
    std::cin >> a;
    std::cout << "Inserire il secondo termine: ";
    std::cin >> b;
    auto r = sum(a, b);
    std::cout << "Il risultato è: " << r << '\n';
}*/

TEST_CASE("Testing sum")
{
    CHECK(0 + 5 == 5);
    CHECK(0 + 0 == 0 );
    CHECK(0 + -1 == -1);
    CHECK(-1 + 1 == 0);
    CHECK(2 + 3 == 5);
    CHECK(3 + 2 == 5);
}

