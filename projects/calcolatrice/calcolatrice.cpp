#include <iostream>
#include <cmath>

double sum(double a, double b)
{
    double r = a + b;
    return r;
}

double sub(double a, double b)
{
    double r = a - b;
    return r;
}

double molt(double a, double b)
{
    double r = a * b;
    return r;
}

double div(double a, double b)
{
    if (a == 0)
    {
        return 0;
    }
    else
    {
        double r = a / b;
        return r;
    }
}

int fact (int a)
{
    int r = 1;
    for (int i = 1; i <= a; i++)
    {
        r = r * i;
    }
    return r;
}

int main()
{
    std::cout << "Inserire il primo termine: ";
    double a;
    std::cin >> a;
    std::cout << "Inserire l'operatore: ";
    char o;
    std::cin >> o;
    if (o != '+' && o != '-' && o != 'x' && o != '/' && o != '^' && o != '!' && o != 'l')
    {
        std::cout << "Inserire uno degli operatori supportati: +, -, x, /, ^, !, l" << '\n';
    }
    if (o == '!')
    {
        int r = fact(a);
        std::cout << a << '!' << '=' << r << '\n';
    }
    if ( o == 'l')
    {
        double r = log(a);
        std::cout << "log" << '(' << a << ')' << '=' << r << '\n';
    }

    if (o != 'l' && o != '!')
    {
        std::cout << "Inserire il secondo termine: ";
        double b;
        std::cin >> b;
        if (o == '+')
        {
            double r = sum(a, b);
            std::cout << a << '+' << b << '=' << r << '\n';
        }
        if (o == '-')
        {
            double r = sub(a, b);
            std::cout << a << '-' << b << '=' << r << '\n';
        }
        if (o == 'x')
        {
            double r = molt(a, b);
            std::cout << a << 'x' << b << '=' << r << '\n';
        }
        if (o == '/')
        {
            if (a != 0 && b == 0)
            {
                std::cout << "Non è possibile dividere per 0" << '\n';
            }
            else
            {
                double r = div(a, b);
                std::cout << a << '/' << b << '=' << r << '\n';
            }
        }
        if (o == '^')
        {
            double r = pow(a, b);
            std::cout << a << '^' << b << '=' << r << '\n';
        }
    }
}