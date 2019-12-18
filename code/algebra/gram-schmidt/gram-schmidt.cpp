#include <cmath>
#include <iostream>

// definizione classe vector

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

vector vect(vector const &l, vector const &r)
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
    return vector{x, y, z};
}

// Algoritmo di Gram-Schmidt

double fourier(vector const &v1, vector const &v2)
{
    double num = v1 * v2;
    double den = v2 * v2;
    return num / den;
}

vector proj(vector const &w1, vector const &w2)
{
    vector proj = fourier(w2, w1) * w1;
    return proj;
}

class Result
{
private:
    vector a;
    vector b;
    vector c;

public:
    Result(vector a, vector b, vector c) : a{a}, b{b}, c{c} {}
    vector v1() const { return a; }
    vector v2() const { return b; }
    vector v3() const { return c; }
};

Result GS(vector const &w1, vector const &w2, vector const &w3)
{
    vector v1 = w1;
    vector v2 = w2 - proj(v1, w2);
    vector v3 = w3 - proj(v1, w3) - (v2, w3);
    Result r = {v1, v2, v3};
    return r;
}

int main()
{
    vector v1 = {1, 2, 0};
    vector v2 = {0, 5, 1};
    vector v3 = {2, 3, 1};
    Result r = GS(v1, v2, v3);
    std::cout << r.v1().x1() << ',' << r.v1().x2() << ',' << r.v1().x3() << '\n';
    std::cout << r.v2().x1() << ',' << r.v2().x2() << ',' << r.v2().x3() << '\n';
    std::cout << r.v3().x1() << ',' << r.v3().x2() << ',' << r.v3().x3() << '\n';
}