#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

bool is_prime(int const &n)
{
    int m;
    int i = 2;
    if (n < 0)
    {
        m = -n;
    }
    else
    {
        m = n;
    }
    while (i < m)
    {
        if (m % i == 0)
        {
            return false;
        }
        else
        {
            i++;
        }
    }
    if (n == 0)
    {
        return false;
    }
    else
    {
        return true;
    }
}

TEST_CASE("Testing is_prime")
{
    CHECK(is_prime(1) == true);
    CHECK(is_prime(10) == false);
    CHECK(is_prime(11) == true);
    CHECK(is_prime(0) == false);
    CHECK(is_prime(2) == true);
    CHECK(is_prime(-7) == true);
    CHECK(is_prime(-9) == false);
}