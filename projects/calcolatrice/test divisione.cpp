#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

double div(double a, double b)
{
    if (a == 0)
    {
        return 0;
    }
    else
    {
        double r = a / b;
        return r;
    }
}

TEST_CASE("testing div")
{
    CHECK(div(2, 3) == 0.667);
    CHECK(div(0, 0) == 0);
    CHECK(div(1, 0) == 12);
    CHECK(div(-1, 1) == -1);
    CHECK(div(4, 2) == 2);
}