#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

class Complex // Forma cartesiana
{
private:
    double r;
    double i;

public:
    /* Complex(double x, double y) : r{x}, i{y} {}
    Complex(double x) : r{x}, i{0.} {}                   // Modi diversi per dichiarare un complesso
    Complex() : r{0.}, i{0.} {}
    */
    Complex(double x = 0, double y = 0) : r{x}, i{y} {}
    double real() const { return r; }
    double imag() const { return i; }
};

// Funzioni

double norm(Complex a)
{
    return (a.real() * a.real()) + (a.imag() * a.imag());
}

Complex conj(Complex a)
{
    return Complex{a.real(), -a.imag()};
}

// Operatori

bool operator==(Complex const &l, Complex const &r)
{
    return l.real() == r.real() && l.imag() == r.imag();
}

bool operator!=(Complex const &l, Complex const &r)
{
    return l.real() != r.real() or l.imag() != r.imag();
}

Complex operator+(Complex const &l, Complex const &r)
{
    return Complex{l.real() + r.real(), l.imag() + r.imag()};
}

Complex operator-(Complex const &l, Complex const &r)
{
    return Complex{l.real() - r.real(), l.imag() - r.imag()};
}

auto operator*(Complex const &l, Complex const &r)
{
    if (l == conj(r))
    {
        double c = norm(l) * norm(l);
        return c;
    }

    if (l != conj(r))
    {
        return Complex{l.real() * r.real(), l.imag() * r.imag()};
    }
}

Complex operator/(Complex const &l, Complex const &r)
{
    if (r.real() != 0 && r.imag() != 0)
    {
        return Complex{l.real() / r.real(), l.imag() / r.imag()};
    }
    else
        return 0;
}



TEST_CASE("Testing Complex")
{
    Complex c0 = {};
    Complex c1 = {1, 1};
    Complex c2 = {2, 2};
    Complex c3 = {3, 3};
    Complex c4 = {4, 4};
    Complex c6 = {6, 6};
    Complex c7 = {-4, -4};
    CHECK(c0 + c1 == c1);
    CHECK(c1 + c2 == c3);
    CHECK(c3 - c2 == c1);
    CHECK(c6 - c2 == c4);
    CHECK(c2 - c6 == c7);
    CHECK(c0 * c2 == c0);
    CHECK(c2 * c3 == c6);
    CHECK(c6 / c1 == c6);
    CHECK(c6 / c3 == c2);
    CHECK(c6 / c2 == c3);
    CHECK(c6 / c0 == 0);
}

TEST_CASE("Testing functions")
{
    Complex c0 = {};
    Complex c1 = {1, 1};
    Complex c2 = {2, 2};
    Complex c3 = {3, 3};
    Complex c4 = {1, -1};
    Complex c5 = {2, -2};
    CHECK(norm(c0) == 0);
    CHECK(norm(c1) == 2);
    CHECK(norm(c2) == 8);
    CHECK(norm(c3) == 18);
    CHECK(conj(c0) == c0);
    CHECK(conj(c1) == c4);
    CHECK(conj(c2) == c5);
}