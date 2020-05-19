#include <thread>
#include <chrono>
#include <random>
#include <iostream>
#include "board.hpp"
#include "display.hpp"

int main()
{
    int const N = 150;
    Board test(N);
    Display display{N};
    display.draw(test);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> infected(0., N);
    for (int i = 0; i != N / 10; ++i)
    {
        test(infected(gen), infected(gen)) = State::Infect;
    }

    std::cout << "Press any key to start the simulation. \n";
    display.WaitKeyPressed();
    for (int i = 0; i != 100; ++i)
    {
        test = evolve(test);
        display.draw(test);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    std::cout << "Press any key to close the window. \n";
    display.WaitKeyPressed();

    /*
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> infected(0., N);
    for (int i = 0; i != N / 10; ++i)
    {
        test(infected(gen), infected(gen)) = State::Infect;
    }
    for (int i = 0; i != 100; ++i)
    {
        test.print();
        test = evolve(test);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    */
}
