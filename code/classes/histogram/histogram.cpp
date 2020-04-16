#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <algorithm>
#include <random>

class Hist1D
{
private:
    int m_nbin;
    std::vector<int> m_bin;
    double m_xlow;
    double m_xup;
    double m_underflow;
    double m_overflow;

    int m_bin_number(double const &x) const
    {
        int const ibin = (m_nbin * (x - m_xlow) / (m_xup - m_xlow));
        assert(ibin >= 0 && ibin < m_bin.size());
        return ibin;
    }

    auto max() const
    {
        return *std::max_element(m_bin.begin(), m_bin.end());
    }

    auto min() const
    {
        return *std::min_element(m_bin.begin(), m_bin.end());
    }

public:
    Hist1D(int nbin, double xlow, double xup) : m_nbin{nbin}, m_xlow{xlow}, m_xup{xup}, m_bin(nbin), m_underflow{0}, m_overflow{0}
    {
        assert(nbin > 0);
    }

    void fill(double const &x)
    {
        if (x < m_xlow)
        {
            ++m_underflow;
        }
        if (x >= m_xup)
        {
            ++m_overflow;
        }
        else
        {
            ++m_bin[m_bin_number(x)];
        }
    }

    void draw() const
    {
        int max_value = max();
        for (int current = max_value; current > 0; --current)
        {
            for (int value : m_bin)
            {
                if (current <= value)
                {
                    std::cout << " * ";
                }
                else
                {
                    std::cout << "   ";
                }
            }
            std::cout << '\n';
        }
        std::cout << "Underflow: " << m_underflow << '\n';
        std::cout << "Overflow : " << m_overflow << '\n';
    }

    /*    void drawh() const
    {
        for (auto it = m_bin.begin(), end = m_bin.end(); it != end; ++it)
        {
            for (int i = 0; i != *it; ++i)
            {
                std::cout << '*';
            }
            std::cout << '\n';
        }
    }*/
};

int main()
{
    double const nbin = 15;
    double const xlow = -2.;
    double const xup = 12.;
    double const mean = 5.;
    double const std = 2.;
    Hist1D h(nbin, xlow, xup);
    // random generation
    std::mt19937 gen;
    std::normal_distribution<double> dist{mean, std};
    for (int n = 0; n != 100; ++n)
    {
        h.fill(dist(gen));
    }
    h.draw();
}