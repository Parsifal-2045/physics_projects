#include <iostream>
#include <stdexcept>

class DivideByZero
{
};

struct InvalidOp
{
    char op;
};

double compute(char op, double l, double r)
{
    double result;

    switch (op)
    {
    case '+':
        result = l + r;
        break;
    case '-':
        result = l - r;
        break;
    case '*':
        result = l * r;
        break;
    case '/':
        if (r == 0)
        {
            throw DivideByZero{};
        }
        else
        {
            result = l / r;
        }
    default:
        throw InvalidOp{op};
        break;
    }
}

int main()
{
    char op;
    double l;
    double r;
    std::cout <<"Inserire l'operazione, premere invio per confermare" << '\n';
    while (std::cin >> l >> op >> r)
    {
        try
        {
            double result = compute(op, l, r);
            std::cout << l << op << r << '=' << result << '\n';
        }
        catch (DivideByZero &)
        {
            std::cerr << "Error: you cannot divide by 0" << '\n';
        }
        catch (InvalidOp const &e)
        {
            std::cerr << "Error: invalid operand " << e.op << '\n';
        }
    }
}