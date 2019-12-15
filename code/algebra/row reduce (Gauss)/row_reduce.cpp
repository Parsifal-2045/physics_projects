#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <iostream>
#include <doctest.h>

class matrix
{
private:
    double a;
    double b;
    double c;
    double d;

public:
    matrix(double a, double b, double c, double d) : a{a}, b{b}, c{c}, d{d} {}
    double a11() const { return a; }
    double a12() const { return b; }
    double a21() const { return c; }
    double a22() const { return d; }
};

bool operator==(matrix const &A, matrix const &B)
{
    return A.a11() == B.a11() && A.a12() == B.a12() && A.a21() == B.a21() && A.a22() == B.a22();
}

matrix reduce(matrix const &D)
{
    if (D.a11() == 0)
    {
    }
    if (D.a11() != 0)
    {
        while (D.a21() != 0)
        {
            double a = (D.a11() / D.a11());
            double b = (D.a12() / D.a11());
            double c = D.a21();
            double d = D.a22();
            c = c - a;
            d = d - b;
            matrix result = {a, b, c, d};
            return result;
        }
    }
}


TEST_CASE("Testing Gauss row reduction")
{
    matrix O = {0,0,0,0};
    matrix m1 = {1,1,1,2};
    CHECK(reduce(m1) == matrix {1,1,0,1});
}