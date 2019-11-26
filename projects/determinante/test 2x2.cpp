#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

int det2(int a, int b, int c, int d)
{
    int r = (a*d) - (b*c);
    return r;
}


TEST_CASE("testing det2") {
    CHECK(det2(1, 0, 0, 1) == 1);
    CHECK(det2(0, 0, 0, 0) == 0);
    CHECK(det2(2, 3, 4, 5) == -2);
}