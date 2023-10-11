#include <iostream>
#include <cassert>
#include <chrono>

template <typename Vector>
class VectorIterator
{
public:
    using ValueType = typename Vector::ValueType;
    using PointerType = ValueType *;
    using ReferenceType = ValueType &;

    VectorIterator(PointerType ptr) : ptr_(ptr) {}

    VectorIterator &operator++()
    {
        ptr_++;
        return *this;
    }

    VectorIterator operator++(int)
    {
        VectorIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    VectorIterator &operator--()
    {
        ptr_--;
        return *this;
    }

    VectorIterator operator--(int)
    {
        VectorIterator tmp = *this;
        --(*this);
        return tmp;
    }

    ReferenceType operator[](int index)
    {
        return *(ptr_[index]);
    }

    PointerType operator->()
    {
        return ptr_;
    }

    ReferenceType operator*()
    {
        return *ptr_;
    }

    bool operator==(VectorIterator const &other) const
    {
        return ptr_ == other.ptr_;
    }

    bool operator!=(VectorIterator const &other) const
    {
        return !(*this == other);
    }

private:
    PointerType ptr_ = nullptr;
};

template <typename T>
class Vector
{
public:
    using ValueType = T;
    using Iterator = VectorIterator<Vector<T>>;

    Vector()
    {
        ReAlloc(2);
    }
    ~Vector()
    {
        Clear();
        ::operator delete[](data_, capacity_ * sizeof(T));
    }

    void PushBack(T const &value)
    {
        if (size_ >= capacity_)
        {
            ReAlloc(capacity_ + capacity_ / 2); // memory allocation size grows geometrically
        }
        new (&data_[size_]) T(value);
        size_++;
    }

    void PushBack(T &&value)
    {
        if (size_ >= capacity_)
        {
            ReAlloc(capacity_ + capacity_ / 2); // memory allocation size grows geometrically
        }
        new (&data_[size_]) T(std::move(value));
        size_++;
    }

    template <typename... Args>
    T &EmplaceBack(Args &&...args)
    {
        if (size_ >= capacity_)
        {
            ReAlloc(capacity_ + capacity_ / 2); // memory allocation size grows geometrically
        }
        new (&data_[size_]) T(std::forward<Args>(args)...); // Construction of object in place
        size_++;
        return data_[size_];
    }

    void PopBack()
    {
        if (size_ > 0)
        {
            size_--;
            data_[size_].~T();
        }
    }

    void Clear()
    {
        for (size_t i = 0; i != size_; i++)
        {
            data_[i].~T();
        }
        size_ = 0;
    }

    T &operator[](size_t index)
    {
        assert(index <= size_);
        return data_[index];
    }

    const T &operator[](size_t index) const
    {
        assert(index <= size_);
        return data_[index];
    }

    size_t Size() const
    {
        return size_;
    }

    Iterator begin()
    {
        return Iterator(data_);
    }

    Iterator end()
    {
        return Iterator(data_ + size_);
    }

private:
    void ReAlloc(size_t new_capacity)
    {
        // 1. allocate a new block of memory
        T *new_block = reinterpret_cast<T *>(::operator new(new_capacity * sizeof(T)));
        if (new_capacity < size_)
        {
            size_ = new_capacity;
        }
        // 2. move all of the old elements in the new block of memory
        for (size_t i = 0; i < size_; i++)
        {
            new (&new_block[i]) T(std::move(data_[i])); // no memcpy, need to call the move constructor for non-primitive types
        }
        for (size_t i = 0; i != size_; i++)
        {
            data_[i].~T();
        }
        // 3. deallocate the old block of memory
        ::operator delete[](data_, capacity_ * sizeof(T));
        data_ = new_block;
        capacity_ = new_capacity;
    }

    T *data_ = nullptr;
    size_t size_ = 0;     // number of elements inside the vector
    size_t capacity_ = 0; // allocated memory -> allocate more than needed to avoid reallocating at each pushback
};

struct Vector3
{
    float x = 0;
    float y = 0;
    float z = 0;
    int *memory_block;

    Vector3() { memory_block = new int[5]; }
    Vector3(float scalar) : x(scalar), y(scalar), z(scalar) { memory_block = new int[5]; }
    Vector3(float x, float y, float z) : x(x), y(y), z(z) { memory_block = new int[5]; }

    Vector3(Vector3 const &other) : x(other.x), y(other.y), z(other.z) {}

    Vector3(Vector3 &&other) : x(other.x), y(other.y), z(other.z) {}

    Vector3 &operator=(Vector3 const &other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }

    Vector3 &operator=(Vector3 &&other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }

    ~Vector3()
    {
        delete[] memory_block;
    }
};

template <typename T>
void PrintVector(Vector<T> &vector)
{
    std::cout << "Not using iterators:\n";

    for (size_t i = 0; i < vector.Size(); i++)
    {
        std::cout << vector[i] << std::endl;
    }
    std::cout << "-----------------------------------------\n";

    std::cout << "Range-based for loop:\n";
    for (T const &value : vector)
    {
        std::cout << value << std::endl;
    }
    std::cout << "-----------------------------------------\n";

    std::cout << "Iterators:\n";
    for (auto it = vector.begin(); it != vector.end(); it++)
    {
        std::cout << *it << std::endl;
    }
    std::cout << "-----------------------------------------\n";
}

template <>
void PrintVector(Vector<Vector3> &vector)
{
    std::cout << "Not using iterators:\n";

    for (size_t i = 0; i < vector.Size(); i++)
    {
        std::cout << vector[i].x << ", " << vector[i].y << ", " << vector[i].z << std::endl;
    }
    std::cout << "-----------------------------------------\n";

    std::cout << "Range-based for loop:\n";
    for (Vector3 const &value : vector)
    {
        std::cout << value.x << ", " << value.y << ", " << value.z << std::endl;
    }
    std::cout << "-----------------------------------------\n";

    std::cout << "Iterators:\n";
    for (auto it = vector.begin(); it != vector.end(); it++)
    {
        std::cout << it->x << ", " << it->y << ", " << it->z << std::endl;
    }
    std::cout << "-----------------------------------------\n";
}

int main()
{
    auto begin = std::chrono::high_resolution_clock::now();
    Vector<Vector3> vector;
    Vector3 vector3(1, 2, 3);
    vector.PushBack(vector3);
    vector.EmplaceBack(4.f, 5.f, 6.f);
    vector.EmplaceBack(4.f, 5.f, 6.f);
    vector.EmplaceBack(4.f, 5.f, 6.f);
    vector.EmplaceBack(4.f, 5.f, 6.f);
    vector.PopBack();
    PrintVector(vector);
    vector.Clear();
    PrintVector(vector);
    vector.EmplaceBack(4.f, 5.f, 6.f);
    PrintVector(vector);

    Vector<std::string> string_vector;
    std::string world = "World";
    string_vector.EmplaceBack("Hello");
    string_vector.PushBack(world);
    PrintVector(string_vector);
    string_vector.PopBack();
    PrintVector(string_vector);
    string_vector.EmplaceBack("!");
    PrintVector(string_vector);
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = end - begin;
    auto time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(diff).count()) / 1e6;
    std::cout << "Time taken: " << time << std::endl;
}