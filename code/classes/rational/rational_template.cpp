#include <stdexcept>
#include <iostream>
#include <numeric>

template <typename ND>
class rational
{
private:
    static_assert(std::is_integral<ND>::value);
    ND n_num;
    ND n_den;

public:
    rational(ND num = 0, ND den = 1) : n_num{num}, n_den{den}
    {
        if (den == 0)
        {
            throw std::runtime_error{"Denominator is zero"};
        }

        if (den < 0)
        {
            num = -num;
            den = -den;
        }

        auto const g = std::gcd(n_num, n_den);
        n_num /= g;
        n_den /= g;
    }
    auto num() const { return n_num; }
    auto den() const { return n_den; }
};

// Operatori

template <typename ND>
bool operator==(rational<ND> const &l, rational<ND> const &r)
{
    return l.num() == r.num() && l.den() == r.den();
}

template <typename ND>
bool operator!=(rational<ND> const &l, rational<ND> const &r)
{
    return !(l == r);
}

template <typename ND>
rational<ND> operator+(rational<ND> const &l, rational<ND> const &r)
{
    if (l.den() == r.den())
    {
        return rational<ND>{l.num() + r.num(), l.den()};
    }
    else
    {
        auto m = std::lcm(l.den(), r.den());
        auto n1 = l.num() * (m / l.den());
        auto n2 = r.num() * (m / r.den());
        return rational<ND>{n1 + n2, m};
    }
}

template <typename ND>
rational<ND> operator-(rational<ND> const &l, rational<ND> const &r)
{
    if (l.den() == r.den())
    {
        return rational<ND>{l.num() - r.num(), l.den()};
    }
    else
    {
        auto m = std::lcm(l.den(), r.den());
        auto n1 = l.num() * (m / l.den());
        auto n2 = r.num() * (m / r.den());
        return rational<ND>{n1 - n2, m};
    }
}

template <typename ND>
rational<ND> operator*(rational<ND> const &l, rational<ND> const &r)
{
    return rational<ND>{l.num() * r.num(), l.den() * r.den()};
}

template <typename ND>
rational<ND> operator/(rational<ND> const &l, rational<ND> const &r)
{
    return rational<ND>{l.num() * r.den(), l.den() * r.num()};
}

//Having inputs separated by /

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

template <typename ND>
auto compute(rational<ND> const &l, char const &op, rational<ND> const &r)
{
    switch (op)
    {
    case '+':
        return l + r;
        break;

    case '-':
        return l - r;
        break;

    case '*':
        return l * r;
        break;

    case '/':
        return l / r;
        break;

    default:
        throw std::runtime_error{"The operator is not supported"};
        break;
    }
}

int main()
{
    try
    {
        std::cout << "Inserire la prima frazione: ";
        long int n1;
        long int d1;
        std::cin >> n1 >> optional_extract('/') >> d1;
        rational a = {n1, d1};
        std::cout << "Inserire l'operatore: ";
        char op;
        std::cin >> op;
        std::cout << "Inserire la seconda frazione: ";
        long int n2;
        long int d2;
        std::cin >> n2 >> optional_extract('/') >> d2;
        rational b = {n2, d2};
        auto r = compute(a, op, b);
        std::cout << a.num() << '/' << a.den() << op << b.num() << '/' << b.den() << '=' << r.num() << '/' << r.den() << '\n';
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}