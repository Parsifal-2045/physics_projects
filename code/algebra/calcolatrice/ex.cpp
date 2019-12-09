#include <iostream>

/*class Error
{
    char n_op;
    double n_l;
    double n_r;

public:
    Error(char op, double l, double r) : n_op{op}, n_l{l}, n_r{r} {}
    char op() const { return n_op; }
    double l() const { return n_l; }
    double r() const { return n_r; }
};*/

class DivideByZero
{
};
struct InvalidOperator
{
    char op;
};

double compute(char op, double const &l, double const &r)
{
    if (op == '+')
    {
        return l + r;
    }
    if (op == '-')
    {
        return l - r;
    }
    if (op == '*')
    {
        return l * r;
    }
    if (op == '/')
    {
        if (r == 0)
        {
            throw DivideByZero{};
        }
        else
        {
            return l / r;
        }
    }
    else
    {
        throw InvalidOperator{op};
    }
}

int main()
{
    char op;
    double l;
    double r;
    while (std::cin >> l >> op >> r)
    {
        try
        {
            double result = compute(op, l, r);
            std::cout << l << op << r << '=' << result << '\n';
        }
        catch (DivideByZero &)
        {
            std::cerr << "Error: cannot divide by 0" << '\n';
        }
        catch (InvalidOperator const &e)
        {
            std::cerr << "Error: invalid operand " << e.op << '\n';
        }
    }
}