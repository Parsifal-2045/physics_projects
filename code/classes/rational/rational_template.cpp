#include <stdexcept>

template <typename ND>
class rational
{
private:
    static_assert(std::is_integral<ND>);  // assert per verificare che numeratore e denominatore siano interi
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
rational<ND> operator+(rational<ND> const &l, rational <ND> const &r)
{
    if (l.den() == r.den())
    {
        return rational<ND>{l.num() + r.num(), l.den()};
    }
    else 
    {
        auto m = lcm(l.den(), r.den());
        auto n1 = l.num() * (m / l.den());
        auto n2 = r.num() * (m / r.den());
        return rational<ND>{n1 + n2, m};
    }
}

template <typename ND>
rational<ND> operator-(rational<ND> const &l, rational <ND> const &r)
{
    if (l.den() == r.den())
    {
        return rational<ND>{l.num() - r.num(), l.den()};
    }
    else 
    {
        auto m = lcm(l.den(), r.den());
        auto n1 = l.num() * (m / l.den());
        auto n2 = r.num() * (m / r.den());
        return rational<ND>{n1 -n2, m};
    }
}

template <typename ND>
rational<ND> operator*(rational<ND> const &l, rational <ND> const &r)
{
    return rational<ND>{l.num() * r.num(), l.den() * r.den()};
}

template <typename ND>
rational<ND> operator/(rational<ND> const &l, rational <ND> const &r)
{
    return rational<ND>{l.num() * r.den(), l.den() * r.num()};
}

int main()
{
    rational<long int> a;
    rational<short int> b;
}