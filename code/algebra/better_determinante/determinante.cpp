#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>

class Matrix
{
private:
    int size_;
    std::vector<double> matrix_;

public:
    Matrix(int n) : size_{n}, matrix_(size_ * size_) {}
    Matrix(std::vector<double> v) : size_{std::sqrt(v.size())}, matrix_(v.size())
    {
        assert(v.size() == matrix_.size());
        std::copy(v.begin(), v.end(), matrix_.begin());
    }
    int size() const { return size_; }
    std::vector<double> matrix() const { return matrix_; }
    auto GetValue(int i, int j) const
    {
        assert(i >= 0 && i < size_ && j >= 0 && j < size_);
        return matrix_[i * size_ + j];
    }
    void print()
    {
        for (int i = 0; i != size_; i++)
        {
            for (int j = 0; j != size_; j++)
            {
                std::cout << GetValue(i, j) << " ";
            }
            std::cout << '\n';
        }
    }
};

double det(Matrix const &M)
{
    auto const n = M.size();
    auto v = M.matrix();
    switch (M.size())
    {
    case 1:
        return v[0];
        break;
    case 2:
        return (v[0] * v[3]) - (v[1] * v[2]);
        break;
    default:
        std::vector<double> vtemp = v;
        vtemp.erase(vtemp.begin(), vtemp.begin() + std::sqrt(vtemp.size()));
        for (auto it = vtemp.begin(); it != vtemp.end(); it += std::sqrt(vtemp.size()))
        {
            vtemp.erase(it);
        }
        Matrix Mtemp{vtemp};
        if (v[0] == 0)
        {
            return 0;
        }
        else
        {
            return v[0] * det(Mtemp);
        }
        break;
    }
}

int main()
{
    std::vector<double> v(4);
    std::iota(v.begin(), v.end(), 0);
    Matrix A{v};
    A.print();
    std::vector<double> v2(9);
    std::iota(v2.begin(), v2.end(), 1);
    Matrix B{v2};
    B.print();
    std::cout << det(A) << " " << det(B) << '\n';

    /*
    v2.erase(v2.begin(), v2.begin() + std::sqrt(v2.size()));
    for (auto it = v2.begin(); it != v2.end(); it += sqrt(v2.size()))
    {
        v2.erase(it);
    }
    Matrix C{v2};
    C.print();
    */
}