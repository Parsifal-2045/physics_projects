#ifndef REGRESSION_HPP
#define REGRESSION_HPP

#include <vector>
#include <cmath>
#include "point.hpp"

struct Result
{
    double A;
    double B;
    double r;
};

bool operator==(Result const &lhs, Result const &rhs)
{
    return lhs.A == rhs.A && lhs.B == rhs.B && lhs.r == rhs.r;
}

bool operator!=(Result const &lhs, Result const &rhs)
{
    return !(lhs == rhs);
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
        for (auto const &end = m_points.end(); remover != end; ++remover)
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
        double const x_med = sum_x / N;
        double const y_med = sum_y / N;
        double num_r = 0;
        double diff_x2 = 0;
        double diff_y2 = 0;
        for (auto it = m_points.begin(), end = m_points.end(); it != end; ++it)
        {
            auto const &v = *it;
            num_r += ((v.x - x_med) * (v.y - y_med));
            diff_x2 += ((v.x - x_med) * (v.x - x_med));
            diff_y2 += ((v.y - y_med) * (v.y - y_med));
        }
        double const num_A = (sum_y * sum_x2) - (sum_x * sum_xy);
        double const num_B = (N * sum_xy) - (sum_x * sum_y);
        double const den_r = sqrt(diff_x2) * sqrt(diff_y2);
        double A = num_A / den;
        double B = num_B / den;
        double r = num_r / den_r;
        return Result{A, B, r};
    }
};

#endif