#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

int det3(int a, int b, int c, int d, int e, int f, int g, int h, int i)
{
    int r = (a*e*i) + (b*f*g) + (c*d*h) - (g*e*c) - (h*f*a) - (i*d*b);
    return r;
}


TEST_CASE("testing det3") {
    CHECK(det3(1, 0, 0, 0, 1, 0, 0, 0, 1) == 1);
    CHECK(det3(0, 0, 0, 0, 0, 0, 0, 0, 0) == 0);
    CHECK(det3(-2, 1, 3, 5, -6, 4, -3, 2, 0) == -20 );
}