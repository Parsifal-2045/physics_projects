#include <cmath>

constexpr bool is_prime(const int n)
{
    if (n > 0 and n <= 2)
    {
        return true;
    }
    else if (n % 2 == 0)
    {
        return false;
    }
    else
    {
        for (int i = 3; i != n; i += 2)
        {
            if (n % i == 0)
            {
                return false;
            }
        }
    }
    return true;
}

int main()
{
    static_assert(!is_prime(10));
    static_assert(!is_prime(4));
    static_assert(is_prime(3));
    static_assert(is_prime(2));
    static_assert(!is_prime(0));
}