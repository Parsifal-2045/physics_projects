#include <iostream>
#include <stdexcept>

double sum(double const &a, double const &b)
{
    return a + b;
}

double sub(double const &a, double const &b)
{
    return a - b;
}

double molt(double const &a, double const &b)
{
    return a * b;
}

double div(double const &a, double const &b)
{
    if (b == 0)
    {
        throw std::runtime_error{"You cannot divide by 0"};
    }
    if (a == 0)
    {
        return 0;
    }
    else
    {
        return a / b;
    }
}

int main()
{
    std::cout << "Inserire il primo termine: ";
    double a;
    std::cin >> a;
    std::cout << "Inserire l'operatore: ";
    char o;
    std::cin >> o;
    std::cout << "Inserire il secondo termine: ";
    double b;
    std::cin >> b;
    try
    {
        if (o == '+')
        {
            return sum(a, b);
        }
        if (0 == '-')
        {
            return sub(a, b);
        }
        if (o == '*')
        {
            return molt(a, b);
        }
        if (o == '/')
        {
            try
            {
                div(a, b);
            }
            catch (std::runtime_error const &e)
            {
                std::cerr << e.what() << '\n';
            }
        }
        else
        {
            throw std::runtime_error{"The inserted operator is not supported"};
        }
    }
    catch (std::runtime_error const &f)
    {
        std::cerr << f.what() << '\n';
    }
}
