#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

double euclideo(double x1, double y1, double z1, double x2, double y2, double z2)
{
    double e = (x1*x2) + (y1*y2) + (z1*z2);
    return e;
}


TEST_CASE("testing euclideo") {
    CHECK(euclideo(0,0,0,0,0,0) == 0);
    CHECK(euclideo(1,1,1,1,1,1) == 3);
    CHECK(euclideo(1,1,1,0,0,0) == 0);
    CHECK(euclideo(1,2,3,1,2,3) == 14);
}