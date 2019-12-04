#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <cmath>

class vector
{
private:
    double x;
    double y;
    double z;

public:
    vector(double x = 0, double y = 0, double z = 0) : x{x}, y{y}, z{z} {}
    double x1() const { return x; }
    double x2() const { return y; }
    double x3() const { return z; }
};

// Operatori

bool operator==(vector const &l, vector const &r)
{
    return l.x1() == r.x1() && l.x2() == r.x2() && l.x3() == r.x3();
}

bool operator!=(vector const &l, vector const &r)
{
    return l.x1() != r.x1() or l.x2() != r.x2() or l.x3() != r.x3();
}

vector operator+(vector const &l, vector const &r)
{
    return vector{l.x1() + r.x1(), l.x2() + r.x2(), l.x3() + r.x3()};
}

vector operator-(vector const &l, vector const &r)
{
    return vector{l.x1() - r.x1(), l.x2() - r.x2(), l.x3() - r.x3()};
}

vector operator*(double const &a, vector const &v)
{
    return vector{a * v.x1(), a * v.x2(), a * v.x3()};
}

vector operator/(vector const &v, double const &a)
{
    return vector{v.x1() / a, v.x2() / a, v.x3() / a};
}

vector operator%(vector const &l, vector const &r)
{
    double a = l.x1();
    double b = l.x2();
    double c = l.x3();
    double d = r.x1();
    double e = r.x2();
    double f = r.x3();
    double x = (b * f) - (c * e);
    double y = (c * d) - (a * f);
    double z = (a * e) - (b * d);
    return vector{x,y,z};
}

double operator*(vector const &l, vector const &r)
{
    double a = l.x1() * r.x1();
    double b = l.x2() * r.x2();
    double c = l.x3() * r.x3();
    double v = a + b + c;
    return v;
}

// Funzioni

double norm(vector const &v)
{
    double r = v * v;
    return sqrt(r);
}

TEST_CASE("Testing vector")
{
    vector v0 = {};
    vector v1 = {1, 1, 1};
    vector v2 = {2, 2, 2};
    CHECK(v0 == vector{0, 0, 0});
    CHECK(v0 != v1);
    CHECK(v1 != vector{1, 1, 2});
    CHECK(v0 + v1 == v1);
    CHECK(v1 + v1 == v2);
    CHECK(v1 + v2 == vector{3, 3, 3});
    CHECK(vector{-1, -1, -1} + v1 == v0);
    CHECK(v0 - v1 == vector{-1, -1, -1});
    CHECK(v2 - v1 == v1);
    CHECK(vector{3, 3, 3} - v2 == v1);
    CHECK(v1 * v0 == 0);
    CHECK(v1 * v2 == 6);
    CHECK(v2 * vector{3, 3, 3} == 18);
    CHECK(vector{1, 0, 1} * v1 == 2);
    CHECK(2 * v1 == v2);
    CHECK(0.5 * v2 == v1);
    CHECK(2 * vector{3, 3, 3} == vector{6, 6, 6});
    CHECK(v2 / 2 == v1);
    CHECK(norm(v0) == 0);
    CHECK(norm (vector{2,0,0}) == 2);
    CHECK(norm(vector{1,0,0} == 1));
    CHECK(vector{1,-1,2} % vector{1,2,3} == vector{-7,-1,3});
    CHECK(v1 % v2 == v0);
}