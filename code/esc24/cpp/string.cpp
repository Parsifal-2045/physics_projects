#include <iostream>
#include <cstring>
#include <memory>

class String
{
    size_t size_;
    std::unique_ptr<char[]> string_;

public:
    // constructor
    String() : size_{}, string_{}
    {
        std::cout << "Default constructor called" << '\n';
    }
    String(char const *s) : size_{std::strlen(s) + 1}, string_{std::make_unique_for_overwrite<char[]>(size_)} // c++20
    {
        std::memcpy(string_.get(), s, size_);
        std::cout << "Constructor called" << '\n';
    }

    // destructor
    ~String() = default;

    // copy constructor
    String(String const &other) : size_{other.size_}, string_{std::make_unique_for_overwrite<char[]>(size_)} // c++20
    {
        std::memcpy(string_.get(), other.string_.get(), size_);
        std::cout << "Copy constructor called" << '\n';
    }

    // copy assignment
    String &operator=(String const &other)
    {
        if (this != &other)
        {
            size_ = other.size_;
            string_ = std::make_unique_for_overwrite<char[]>(size_); // c++20
            std::memcpy(string_.get(), other.string_.get(), size_);
            std::cout << "Copy assignment called" << '\n';
        }
        return *this;
    }

    // move constructor
    String(String &&tmp) noexcept : size_{tmp.size_}
    {
        string_.swap(tmp.string_);
        std::cout << "Move constructor called" << '\n';
    }

    // move assignment
    String &operator=(String &&tmp) noexcept
    {
        size_ = tmp.size_;
        string_.swap(tmp.string_);
        std::cout << "Move assignment called" << '\n';
        return *this;
    }

    void print()
    {
        std::cout << '"';
        for (size_t i = 0; i != size_; ++i)
        {
            std::cout << string_[i];
        }

        std::cout << '"' << '\n';
    }
};

String get_string()
{
    return "Hi!";
}

int main()
{
    std::cout << "\n Try default constructor" << '\n';
    String s_default;
    std::cout << "String: ";
    s_default.print();

    std::cout << "\n Try constructor" << '\n';
    String s{"Test"};
    std::cout << "String: ";
    s.print();

    std::cout << "\n Try copy constructor" << '\n';
    std::cout << "Original string: ";
    s.print();
    String s_copy{s};
    std::cout << "Copied string: ";
    s_copy.print();

    std::cout << "\n Try copy assignment on different strings" << '\n';
    std::cout << "Original string: ";
    s.print();
    s_default = s;
    std::cout << "Assigned string: ";
    s_default.print();

    std::cout << "\n Try copy assignment on same string" << '\n';
    std::cout << "Original string: ";
    s_copy.print();
    s_copy = s_copy;
    std::cout << "Assigned string: ";
    s_copy.print();

    std::cout << "\n Try move constructor" << '\n';
    std::cout << "Starting string: ";
    s.print();
    String s3{std::move(s)};
    std::cout << "Moved string: ";
    s3.print();

    std::cout << "\n Try move assignment" << '\n';
    std::cout << "String before move assignment: " << '\n';
    s3.print();
    s3 = get_string();
    std::cout << "String after move assignment: " << '\n';
    s3.print();
}