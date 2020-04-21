#include <iostream>
#include <stdexcept>
#include "point.hpp"
#include "regression.hpp"
#include "optional_extract.hpp"

int main()
{
    try
    {
        Regression reg;
        int N;
        std::cout << "Declare the number of points to fit: ";
        std::cin >> N;
        if (N < 2)
        {
            throw std::runtime_error{"Not enough points"};
        }
        for (int i = 1; i != N + 1; i++)
        {
            double x;
            double y;
            std::cout << "Insert the coordinates of the point " << i << " separated by a comma: ";
            std::cin >> x >> optional_extract(',') >> y;
            reg.add({x, y});
        }
        auto result = reg.fit();
        std::cout << "Y = " << result.A << " + " << result.B << " X\n";
        std::cout << "The linear correlation coefficient is: " << result.r << '\n';
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}