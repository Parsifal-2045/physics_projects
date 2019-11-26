#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

int isqrt (int a)
{
    int i = 1;
    while (i*i < a) {
        i++;
    } 
    if (i*i > a) {
        i--;
    }
    return i;
}

TEST_CASE("testing isqrt") {
    CHECK(isqrt(0) == 0);
    CHECK(isqrt(9) == 3);
    CHECK(isqrt(10) == 3);
}