#include <cmath>
#include <iostream>
#include <cctype>

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

struct optional_extract
{
    char c;
    optional_extract(char c) : c{c} {}
};

std::istream &operator>>(std::istream &ins, optional_extract e)
{
    // Skip leading whitespace IFF user is not asking to extract a whitespace character
    if (!std::isspace(e.c))
        ins >> std::ws;

    // Attempt to get the specific character
    if (ins.peek() == e.c)
        ins.get();

    // There is no failure!
    return ins;
}

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
    std::cout << "Inserire le coordinate del primo vettore: ";
    double a, b, c;
    std::cin >> a >> optional_extract(',') >> b >> optional_extract(',') >> c;
    vector w1 = {a, b, c};
    std::cout << "Inserire le coordinate del secondo vettore: ";
    double d, e, f;
    std::cin >> d >> optional_extract(',') >> e >> optional_extract(',') >> f;
    vector w2 = {d, e, f};
    std::cout << "Inserire le coordinate del terzo vettore: ";
    double g, h, i;
    std::cin >> g >> optional_extract(',') >> h >> optional_extract(',') >> i;
    vector w3 = {g, h, i};
    Result r = GS(w1, w2, w3);
    std::cout << "La base ortogonale è formata dai seguenti vettori: " << '\n';
    std::cout << "v1 = " << r.v1().x1() << ',' << r.v1().x2() << ',' << r.v1().x3() << '\n';
    std::cout << "v2 = " << r.v2().x1() << ',' << r.v2().x2() << ',' << r.v2().x3() << '\n';
    std::cout << "v3 = " << r.v3().x1() << ',' << r.v3().x2() << ',' << r.v3().x3() << '\n';
}