#include <iostream>

#include "pybind11/include/pybind11/pybind11.h"

namespace py = pybind11;

int add(const int i, const int j) { return i + j; }
float add(const float i, const float j) { return i + j; }

struct testClass
{
    testClass() = default;
    void hello()
    {
        std::cout << "Hello from C++" << '\n';
    }
};

struct testDerived : public testClass
{
    testDerived() = default;
    void goodbye()
    {
        std::cout << "Goodbye from C++" << '\n';
    }
};

PYBIND11_MODULE(TestModule, t)
{
    t.def("add", py::overload_cast<int, int>(&add), py::arg("x"), py::arg("y"));
    t.def("add", py::overload_cast<float, float>(&add), py::arg("x") = 0.f, py::arg("y") = 0.f);
    py::class_<testClass>(t, "testClass")
        .def(py::init<>())
        .def("hello", &testClass::hello);
    py::class_<testDerived, testClass>(t, "testDerived")
        .def(py::init<>())
        .def("goodbye", &testDerived::goodbye);
}