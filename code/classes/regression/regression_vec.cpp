#include <iostream>
#include <vector>
#include <stdexcept>
#include <numeric>

struct Point
{
    double x;
    double y;
};

bool operator==(Point const &lhs, Point const &rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

bool operator!=(Point const &lhs, Point const &rhs)
{
    return !(lhs == rhs);
}

Point operator+(Point const &lhs, Point const &rhs)
{
    return Point{lhs.x + rhs.x, lhs.y + rhs.y};
}

struct Result
{
    double A;
    double B;
};

bool operator==(Result const &l, Result const &r)
{
    return l.A == r.A && l.B == r.B;
}

bool operator!=(Result const &l, Result const &r)
{
    return !(l == r);
}

class Regression
{
private:
    std::vector<Point> m_points;

public:
    void add(double x, double y)
    {
        m_points.push_back({x, y});
    }

    auto remove(double x, double y)
    {
        Point const p = {x, y};
        auto remover = m_points.begin();
        for (auto const& end = m_points.end(); remover != end; ++remover)
        {
            Point const &current = *remover;
            if (current == p)
            {
                break;
            }
        }
        if (remover != m_points.end())
        {
            m_points.erase(remover);
            return true;
        }
        else 
        {
            return false;
        }
    }

    auto fit() const
    {
        int const N = m_points.size();
        if (N < 2)
        {
            throw std::runtime_error{"Not enough points"};
        }
        double sum_x = 0.;
        double sum_y = 0.;
        double sum_x2 = 0.;
        double sum_xy = 0.;
        for (auto it = m_points.begin(), end = m_points.end(); it != end; ++it)
        {
            auto const &v = *it;
            sum_x += v.x;
            sum_y += v.y;
            sum_x2 += v.x * v.x;
            sum_xy += v.x * v.y;
        }
        double const den = (N * sum_x2) - (sum_x * sum_x);
        if (den == 0)
        {
            throw std::runtime_error{"Vertical line"};
        }
        double const num_A = (sum_y * sum_x2) - (sum_x * sum_xy);
        double const num_B = (N * sum_xy) - (sum_x * sum_y);
        double A = num_A / den;
        double B = num_B / den;
        return Result{A, B};
    }
};

int main()
{
    try
    {
        Regression reg;
        reg.add(0, 0);
        reg.add(1, 1);
        reg.add(2, 2);
        reg.remove(1, 1);
        auto result = reg.fit();
        // should print Y = 0 + 1 X
        std::cout << "Y = " << result.A << " + " << result.B << " X\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}