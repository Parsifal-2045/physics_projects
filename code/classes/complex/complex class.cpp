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

Complex operator*(Complex const &l, Complex const &r)
{
    double real = (l.real() * r.real()) - (l.imag() * r.imag());
    double imag = (l.real() * r.imag()) + (l.imag() * r.real());
    return Complex{real, imag};
}

Complex operator/(Complex const &l, Complex const &r)
{
   double nr = (l.real() * r.real()) + (l.imag() * r.imag());
   double ni = (l.imag() * r.real()) - (l.real() * r.imag());
   double d = (r.real() * r.real()) + (r.imag() * r.imag());
   return Complex{nr / d, ni / d};
}

// Funzioni

double norm(Complex const &a)
{
    return (a.real() * a.real()) + (a.imag() * a.imag());
}

Complex conj(Complex const &a)
{
    return Complex{a.real(), -a.imag()};
}

TEST_CASE("Testing Complex")
{
    Complex c0 = {};
    Complex c1 = {1, 1};
    Complex c2 = {2, 2};
    Complex c3 = {3, 3};
    Complex c4 = {4, 4};
    Complex c6 = {6, 6};
    Complex c7 = {1, -1};
    CHECK(c0 + c1 == c1);
    CHECK(c1 + c2 == c3);
    CHECK(c3 - c2 == c1);
    CHECK(c6 - c2 == c4);
    CHECK(c1 * c7 == Complex{2, 0});
    CHECK(c1 * c2 == Complex{0, 4});
    CHECK(c2 * c3 == Complex{0, 12});
    CHECK(c1 / c7 == Complex{0, 1});
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