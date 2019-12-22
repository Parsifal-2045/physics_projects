#include <iostream>
#include <stdexcept>

//Row class

class row
{
private:
    double x;
    double y;
    double z;

public:
    row(double x, double y, double z) : x{x}, y{y}, z{z} {}
    double x1() const { return x; }
    double x2() const { return y; }
    double x3() const { return z; }
};

//Operators

bool operator==(row const &L, row const &R)
{
    return L.x1() == R.x1() && L.x2() == R.x2() && L.x3() == R.x3();
}

row operator-(row const &U, row const &D)
{
    double x1 = U.x1();
    double x2 = U.x2();
    double x3 = U.x3();
    double y1 = D.x1();
    double y2 = D.x2();
    double y3 = D.x3();
    double r1 = x1 - y1;
    double r2 = x2 - y2;
    double r3 = x3 - y3;
    row result = {r1, r2, r3};
    return result;
}

row operator*(double const &a, row const &R)
{
    double x = R.x1();
    double y = R.x2();
    double z = R.x3();
    row result = {a * x, a * y, a * z};
    return result;
}

//Matrix class

class matrix
{
private:
    row R1;
    row R2;
    row R3;

public:
    matrix(row R1, row R2, row R3) : R1{R1}, R2{R2}, R3{R3} {}
    row r1() const { return R1; }
    row r2() const { return R2; }
    row r3() const { return R3; }
};

//Operators

bool operator==(matrix const &A, matrix const &B)
{
    return A.r1() == B.r1() && A.r2() == B.r2() && A.r3() == B.r3();
}

//Having commas between coordinates

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

//Reduce

matrix reduce(matrix const &A)
{
    if (A.r1().x1() == 0)
    {
        throw std::runtime_error{"The first term cannot be 0, please change the order of the rows"};
    }
    if (A.r1().x1() != 0)
    {
        row R1 = A.r1();
        row R2 = A.r2();
        row R3 = A.r3();
        double c1 = R2.x1() / R1.x1();
        R2 = R2 - (c1 * R1);
        double c2 = R3.x1() / R1.x1();
        R3 = R3 - (c2 * R1);
        if (R2.x2() != 0)
        {
            double c3 = R3.x2() / R2.x2();
            R3 = R3 - (c3 * R2);
            matrix Result = {R1, R2, R3};
            return Result;
        }
        if (R2.x2() == 0 && R3.x2() != 0)
        {
            row r1 = R1;
            row r2 = R3;
            row r3 = R2;
            double c3 = r3.x2() / r2.x2();
            r3 = r3 - (c3 * r2);
            matrix result = {r1, r2, r3};
            return result;
        }
        if (R2.x2() == 0 && R3.x2() == 0)
        {
            matrix r = {R1, R2, R3};
            return r;
        }
    }
}

int main()
{
    std::cout << "Inserire la prima riga della matrice: ";
    double a, b, c;
    std::cin >> a >> optional_extract(',') >> b >> optional_extract(',') >> c;
    row r1 = {a, b, c};
    std::cout << "Inserire la seconda riga della matrice: ";
    double d, e, f;
    std::cin >> d >> optional_extract(',') >> e >> optional_extract(',') >> f;
    row r2 = {d, e, f};
    std::cout << "Inserire la terza riga della matrice: ";
    double g, h, i;
    std::cin >> g >> optional_extract(',') >> h >> optional_extract(',') >> i;
    row r3 = {g, h, i};
    matrix A = {r1, r2, r3};
    try
    {
        matrix R = reduce(A);
        std::cout << "La matrice ridotta in forma scala per righe è: " << '\n';
        std::cout << R.r1().x1() << R.r1().x2() << R.r1().x3() << '\n';
        std::cout << R.r2().x1() << R.r2().x2() << R.r2().x3() << '\n';
        std::cout << R.r3().x1() << R.r3().x2() << R.r3().x3() << '\n';
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << e.what() << '\n';
    }
}