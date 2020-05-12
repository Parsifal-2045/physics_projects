#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <cassert>

template <class T>
class Vector
{
private:
    int size_ = 0;
    T *data_ = nullptr;

public:
    Vector() = default;
    Vector(int n, T const &v = T{}) : size_{n}, data_{new T[size_]}
    {
        std::fill(data_, data_ + size_, v);
    }
    ~Vector() //class destructor avoids memory leaks
    {
        delete[] data_;
    }
    Vector &operator=(Vector const &other)
    {
        if (this != &other)
        {
            delete[] data_;
            size_ = other.size_;
            data_ = new T[size_];
            std::copy(other.data_, other.data_ + size_, data_);
        }
        return *this;
    }
Vector(Vector const &other) : size_{other.size_}, data_{new T[size_]} //copy constructor to avoid double deletion of pointers when calling the destructor on a copied vector
{
    std::copy(other.data_, other.data_ + size_, data_);
}
T &operator[](int i)
{
    assert(i >= 0 && i < size_);
    return data_[i];
}
T operator[](int i) const
{
    assert(i >= 0 && i < size_);
    return data_[i];
}
}
;

TEST_CASE("Testing vector")
{
    Vector<int> vi{};
    Vector<double> vd(16);
    CHECK(vd[8] == 0);
    vd[8] = 42.;
    auto vd2 = vd;
    CHECK(vd2[8] == 42);
    Vector<double> vd3;
    vd3 = vd;
    CHECK(vd3[8] == 42.);
    vd2 = vd;
    vd = vd;
    CHECK(vd[8] == 42.);
}