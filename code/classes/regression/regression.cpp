#include <iostream>
#include <cassert>

struct fit_s
{
    double A;
    double B;
};

class Regression
{
private:
    int m_N;
    double m_sum_x;
    double m_sum_y;
    double m_sum_xy;
    double m_sum_x2;

public:
    Regression() : m_N{0}, m_sum_x{0}, m_sum_y{0}, m_sum_xy{0}, m_sum_x2{0} {}
    void add(double x, double y)
    {
        ++m_N;
        m_sum_x += x;
        m_sum_y += y;
        m_sum_xy += x * y;
        m_sum_x2 += x * x;
    }

    fit_s const fit()
    {
        assert(m_N>1);
        double const num_A = (m_sum_y * m_sum_x2) - (m_sum_x * m_sum_xy);
        double const num_B = (m_N * m_sum_xy) - (m_sum_x * m_sum_y);
        double const den = (m_N * m_sum_x2) - (m_sum_x * m_sum_x);
        double A = num_A / den;
        double B = num_B / den;
        fit_s result{A,B};
        return result;
    }
};

int main()
{
    Regression reg;
    reg.add(0, 0);
    reg.add(1, 1);
    auto result = reg.fit();
    std::cout << "Y = " << result.A << " + " << result.B << " X\n";
}
