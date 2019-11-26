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

Complex operator+(Complex const &a, Complex const &b)
{
    return Complex{a.real() + b.real(), a.imag() + b.imag()};
}

int main()
{
    Complex a = {1, 2};
    Complex b = {2, 3};
    a + b;
}