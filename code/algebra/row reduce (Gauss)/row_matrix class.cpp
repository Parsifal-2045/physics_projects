
#include <iostream>

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

bool operator==(matrix const &A, matrix const &B)
{
    return A.r1() == B.r1() && A.r2() == B.r2() && A.r3() == B.r3();
}

matrix reduce(matrix const &D)
{
    if (D.r1().x1() == 0)
    {
    }
    if (D.r1().x1() != 0)
    {
        double a = 1;
        double b = D.r1().x2() / D.r1().x1();
        double c = D.r1().x3() / D.r1().x1();
        row r1 = {a, b, c};
        double d = D.r2().x1();
        double e = D.r2().x2();
        double f = D.r2().x3();
        row r2 = {d, e, f};
        double g = D.r3().x1();
        double h = D.r3().x2();
        double i = D.r3().x3();
        row r3 = {g, h, i};
        while (d != 0)
        {
            row r2 = r2 - r1;
        }
        while (g != 0)
        {
            row r3 = r3 - r1;
        }
        while (h != 0)
        {
            d = 0;
            e = 1;
            f = r2.x3() / r2.x2();
            row r2 = {d, e, f};
            row r3 = r3 - r2;
        }
        matrix R = {r1, r2, r3};
        return R;
    }
}

int main()
{
    row r4 = {1, 1, 1};
    row r5 = {2, 4, 5};
    row r6 = {3, 4, 5};
    matrix B = {r4, r5, r6};
    matrix R = reduce(B);
    std::cout << R.r1().x1() << R.r1().x2() << R.r1().x3() << '\n';
    std::cout << R.r2().x1() << R.r2().x2() << R.r2().x3() << '\n';
    std::cout << R.r3().x1() << R.r3().x2() << R.r3().x3() << '\n';
}

/*TEST_CASE("Testing rows")
{
    row r1 = {1, 0, 0};
    row r2 = {0, 1, 0};
    row r3 = {0, 0, 1};
    matrix A = {r1, r2, r3};
    CHECK(r1 - r2 == row{1, -1, 0});
    CHECK(r1 - r3 == row{1, 0, -1});
    CHECK(reduce(A) == A);
    row r4 = {1, 1, 1};
    row r5 = {2, 4, 5};
    row r6 = {3, 4, 5};
    matrix B = {r4, r5, r6};
    row r7 = {0, 1, 3 / 2};
    row r8 = {0, 0, 1 / 2};
    CHECK(reduce(B) == matrix{r4, r7, r8});
}*/