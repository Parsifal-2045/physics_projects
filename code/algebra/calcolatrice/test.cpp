#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

double molt(double a, double b)
{
    double r = a * b;
    return r;
}

TEST_CASE("testing molt")
{
    CHECK(molt(2, 3) == 6);
    CHECK(molt(0, 0) == 0);
    CHECK(molt(2, 0.5) == 1);
    CHECK(molt(-1, 1) == -1);
    CHECK(molt(-1, -1) == 1);
    CHECK(molt(-1, 3) == -3);
    CHECK(molt(2, 0) == 0);
}

double sum(double a, double b)
{
    double r = a + b;
    return r;
}

TEST_CASE("testing sum")
{
    CHECK(sum(2, 3) == 5);
    CHECK(sum(0, 0) == 0);
    CHECK(sum(2.3, 0.7) == 3);
    CHECK(sum(-1, 1) == 0);
    CHECK(sum(-0.7, -0.3) == -1);
}

int fact (int a)
{
    int r = 1;
    for (int i = 1; i <= a; i++)
    {
        r = r * i;
    }
    return r;
}

TEST_CASE("testing fact")
{
    CHECK(fact(0) == 1);
    CHECK(fact(2) == 2);
    CHECK(fact(5) == 120);
    CHECK(fact(6) == 720);
}