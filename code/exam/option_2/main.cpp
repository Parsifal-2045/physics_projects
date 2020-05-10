#include <random>
#include <thread>
#include <chrono>
#include "board.hpp"

auto evolve(Board const &current)
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0., 1.);
    auto contagion_probability = dis(gen);
    auto heal_probability = dis(gen);
    int const N = current.size();
    Board next(N);
    for (int i = 0; i != N; ++i)
    {
        for (int j = 0; j != N; ++j)
        {
            std::mt19937 status_gen(std::random_device{}());
            std::uniform_real_distribution<> status_dis(0., 1.);
            auto status = status_dis(status_gen);
            if (current.GetCellState(i, j) == State::Recovered)
            {
                next(i, j) = current.GetCellState(i, j);
                assert(next.GetCellState(i, j) == State::Recovered);
            }
            if (current.GetCellState(i, j) == State::Infect)
            {
                if (heal_probability > status)
                {
                    next(i, j) = State::Recovered;
                }
                else
                {
                    next(i, j) = current.GetCellState(i, j);
                    assert(next.GetCellState(i, j) == State::Infect);
                }
            }
            if (current.GetCellState(i, j) == State::Susceptible)
            {
                if (current.CheckNeighbours(i, j) == 0)
                {
                    next(i, j) = current.GetCellState(i, j);
                    assert(next.GetCellState(i, j) == State::Susceptible);
                }
                else
                {
                    int infected = current.CheckNeighbours(i, j);
                    auto contagion = infected * contagion_probability;
                    if (contagion > status)
                    {
                        next(i, j) = State::Infect;
                    }
                    else
                    {
                        next(i, j) = current.GetCellState(i, j);
                        assert(next.GetCellState(i, j) == State::Susceptible);
                    }
                }
            }
        }
    }
    return next;
}

int main()
{
    int const N = 50;
    Board test(N);
    test(0, 0) = State::Infect;
    test(6, 7) = State::Infect;
    test(5, 5) = State::Infect;
    for (int i = 0; i != 100; ++i)
    {
        test.print();
        test = evolve(test);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}
