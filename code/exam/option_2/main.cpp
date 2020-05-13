#include <thread>
#include <chrono>
#include <random>
#include "board.hpp"

int main()
{
    int const N = 50;
    Board test(N);
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
}
