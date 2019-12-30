#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <stdexcept>

int gcd(int a, int b) // Funzione massimo comun divisore
{
    while (a != b)
    {
        if (a > b)
        {
            a -= b; //come scrivere a = a - b
        }
        else
        {
            b -= a; //come scrivere b = b - a
        }
    }
    return a;
}

int mcm(int a, int b) // Funzione minimo comune multiplo
{
    if (a == b)
    {
        return a;
    }
    if (gcd(a, b) == 1)
    {
        int m = a * b;
        return m;
    }
    if (gcd(a, b) != 1)
    {
        int m = (a * b) / gcd(a, b);
        return m;
    }
}

class Rational
{
    int n_num;
    int n_den;

public:
    Rational(int num = 0, int den = 1) : n_num{num}, n_den{den}
    {
        if (den == 0)
        {
            throw std::runtime_error{"denominator is zero"};
        }

        if (den < 0)
        {
            num = -num;
            den = -den;
        }

        auto const g = gcd(n_num, n_den);
        n_num /= g;
        n_den /= g;
    }

    int den() const { return n_den; }
    int num() const { return n_num; }
};

// Operatori

bool operator==(Rational const &l, Rational const &r)
{
    return l.num() == r.num() && l.den() == r.den();
}

bool operator!=(Rational const &l, Rational const &r)
{
    return !(l == r);
}

Rational operator+(Rational const &l, Rational const &r)
{
    if (l.den() == r.den())
    {
        return Rational{l.num() + r.num(), l.den()};
    }
    else
    {
        int m = mcm(l.den(), r.den());
        int n1 = l.num() * (m / l.den());
        int n2 = r.num() * (m / r.den());
        return Rational{n1 + n2, m};
    }
}

Rational operator-(Rational const &l, Rational const &r)
{
    if (l.den() == r.den())
    {
        return Rational{l.num() - r.num(), l.den()};
    }
    else
    {
        int m = mcm(l.den(), r.den());
        int n1 = l.num() * (m / l.den());
        int n2 = r.num() * (m / r.den());
        return Rational{n1 - n2, m};
    }
}

Rational operator*(Rational const &l, Rational const &r)
{
    return Rational{l.num() * r.num(), l.den() * r.den()};
}

Rational operator/(Rational const &l, Rational const &r)
{
    return Rational{l.num() * r.den(), l.den() * r.num()};
}

TEST_CASE("testing Rational")
{
    Rational r1 = {1, 1};
    Rational r2 = {1, 2};
    Rational r3 = {2, 4};
    Rational r4 = {4, 8};
    CHECK(r2 == r4);
    CHECK(r3 == r4);
    CHECK(r2 == r4);
    CHECK(r1 != r2);
    CHECK(r2 + r3 == r1);
    CHECK_THROWS((Rational{0,0}));
    Rational r5 = {2, 3};
    Rational r6 = {7, 6};
    Rational r7 = {1, 6};
    Rational r8 = {1, 4};
    Rational r9 = {1, 3};
    Rational r10 = {4, 3};
    CHECK(r2 + r5 == r6);
    CHECK(r3 + r5 == r6);
    CHECK(r5 - r2 == r7);
    CHECK(r2 * r3 == r8);
    CHECK(r2 * r5 == r9);
    CHECK(r2 * r1 == r2);
    CHECK(r2 / r2 == r1);
    CHECK(r5 / r2 == r10);
}