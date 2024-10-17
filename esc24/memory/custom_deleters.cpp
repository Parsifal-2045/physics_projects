#include <cstdio>
#include <iostream>
#include <memory>

int main() {
    // Custom Destructor struct
    struct Deleter {
        auto operator()(FILE* f) const {
            std::cout << "Custom struct Deleting" << '\n';
            std::fclose(f);
        }
    };
    std::unique_ptr<FILE, Deleter> ptr{std::fopen("/etc/", "r")};

    // Lambda
    using LambdaDeleter = void (*)(FILE*);
    LambdaDeleter ld = [](FILE* f) {
        std::cout << "Lambda Deleting" << '\n';
        std::fclose(f);
    };
    std::unique_ptr<FILE, LambdaDeleter> lptr{std::fopen("/etc/", "r"), ld};

    // Lambda with decltype (C++20)
    auto l = [](FILE* f) {
        std::cout << "Lambda decltype deleting" << '\n';
        std::fclose(f);
    };
    std::unique_ptr<FILE, decltype(l)> ptr1{std::fopen("/etc/", "r")};
}
