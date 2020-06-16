#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <random>
#include <iostream>

//Manipulation of vectors with algorithms, checking random distribution is random

/*
int main()
{
    std::vector<double> v;
    std::vector<int> a{1,1,13,14,1,12,14,1,1};    
    for (int j = 0; j != 10; j++)
    {
        for (int i = 0; i != 10; i++)
        {
            std::mt19937 gen (std::random_device{}());
            std::uniform_real_distribution<> dis (0., 1.);
            auto status = dis(gen);
            v.push_back(status);  

        }
    }
    std::cout << v.size() << '\n';
    std::cout << a.size() << '\n';
    std::sort(v.begin(), v.end());
    std::sort(a.begin(), a.end());
    auto it = std::unique(v.begin(), v.end());
    v.resize(std::distance(v.begin(), it));
    auto it2 = std::unique(a.begin(), a.end());
    a.resize(std::distance(a.begin(), it2));
    std::cout << v.size() << '\n';
    std::cout << a.size() << '\n';
}
*/

// Lambda expressions

/*
int main()
{
    std::vector<int> v(10);
    std::iota(v.begin(), v.end(), 0);
    std::vector<std::string> s(v.size());
    std::string name;
    std::cin >> name;
    std::transform(v.begin(), v.end(), s.begin(), [=](int n) { return name + std::to_string(n); });
    for (auto &str : s)
    {
        std::cout << str << '\n';
    }
}
*/

//Smallest of N numbers

/*
int main()
{
    std::vector<int> v;
    int i;
    while (std::cin >> i)
    {
        v.push_back(i);
    }
    std::sort(v.begin(), v.end());
    std::cout << v[0] << '\n';
}
*/

//Integer square root

/*
int isqrt(int n)
{
    int i = 1;
    while (i * i < n)
    {
        ++i;
    }
    if (i * i > n)
    {
        --i;
    }
    return i;
}

int main()
{
    std::cout << isqrt(25) << '\n';
    std::cout << isqrt(30) << '\n';
    std::cout << isqrt(36) << '\n';
}
*/

//Sum of the first N numbers

/*
int main()
{
    int n;
    std::cin >> n ;
    int sum = 0;
    for(int i = 1; i <= n; i++)
    {
        sum += i;        
    }
    std::cout << sum <<'\n';
}
*/

//Power, gcd, lcm and is_prime

/*
int pow(int b, int e)
{
    int p = 1;
    for (int i = 0; i != e; ++i)
    {
        p = p * b;
    }
    return p;
}

int gcd(int a, int b)
{
    int r;
    while (b != 0)
    {
        r = a % b;
        a = b;
        b = r;
    }
    return a;
}

int lcm(int a, int b)
{
    if (a == 0 || b == 0)
    {
        return 0;
    }
    else
    {
        return (a * b) / gcd(a, b);
    }
}

bool is_prime(int n)
{
    if (n == 0 || n == 1)
    {
        return false;
    }
    for (int i = 2; i <= n / 2; i++)
    {
        if (n % i == 0)
        {
            return false;
        }
    }
    return true;
}
*/

//Pi

/*
double pi()
{
    int const N = 100000;
    double const delta_x = 1. / N;
    double sum = 0.;
    for (int i = 0; i != N; ++i)
    {
        double x = i * delta_x;
        sum += 4 / (1 + x * x);
    }
    return sum * delta_x;
}
*/