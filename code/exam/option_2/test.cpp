#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "board.hpp"

Board EvolveTestContagion(Board const &current)
{
    auto contagion_probability = 1;
    auto heal_probability = 0;
    int const N = current.size();
    Board next(N);
    for (int i = 0; i != N; ++i)
    {
        for (int j = 0; j != N; ++j)
        {
            auto status = 0.5;
            if (current.GetCellState(i, j) == State::Recovered)
            {
                next(i, j) = current.GetCellState(i, j);
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
                }
            }
            if (current.GetCellState(i, j) == State::Susceptible)
            {
                if (current.CheckNeighbours(i, j) == 0)
                {
                    next(i, j) = current.GetCellState(i, j);
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
                    }
                }
            }
        }
    }
    return next;
}

Board EvolveTestHeal(Board const &current)
{
    auto contagion_probability = 0;
    auto heal_probability = 1;
    int const N = current.size();
    Board next(N);
    for (int i = 0; i != N; ++i)
    {
        for (int j = 0; j != N; ++j)
        {
            auto status = 0.5;
            if (current.GetCellState(i, j) == State::Recovered)
            {
                next(i, j) = current.GetCellState(i, j);
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
                }
            }
            if (current.GetCellState(i, j) == State::Susceptible)
            {
                if (current.CheckNeighbours(i, j) == 0)
                {
                    next(i, j) = current.GetCellState(i, j);
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
                    }
                }
            }
        }
    }
    return next;
}

TEST_CASE("Testing spread and heal on board")
{
    int N = 10;
    Board test(N);
    for (int i = 0; i != N; i++)
    {
        for (int j = 0; j != N; j++)
        {
            CHECK(test.GetCellState(i, j) == State::Susceptible);
        }
    }
    CHECK(GetSusceptible(test) == N * N);
    CHECK(test.GetCellState(N + 1, N + 1) == State::Recovered);
    CHECK(test.GetCellState(-1, -1) == State::Recovered);

    SUBCASE("Testing infection rate in the board")
    {
        test(2, 3) = State::Infect;
        CHECK(GetInfected(test) == 1);
        CHECK(GetSusceptible(test) == (N * N) - 1);

        Board evolved_1 = EvolveTestContagion(test);
        CHECK(evolved_1.GetCellState(1, 2) == State::Infect);
        CHECK(evolved_1.GetCellState(1, 3) == State::Infect);
        CHECK(evolved_1.GetCellState(1, 4) == State::Infect);
        CHECK(evolved_1.GetCellState(2, 2) == State::Infect);
        CHECK(evolved_1.GetCellState(2, 3) == State::Infect);
        CHECK(evolved_1.GetCellState(2, 4) == State::Infect);
        CHECK(evolved_1.GetCellState(3, 2) == State::Infect);
        CHECK(evolved_1.GetCellState(3, 3) == State::Infect);
        CHECK(evolved_1.GetCellState(3, 4) == State::Infect);
        CHECK(GetInfected(evolved_1) == 9);
        CHECK(GetSusceptible(evolved_1) + GetInfected(evolved_1) + GetRecovered(evolved_1) == N * N);

        Board evolved_2 = EvolveTestContagion(evolved_1);
        for (int i = 0; i != 5; i++)
        {
            for (int j = 1; j != 6; j++)
            {
                CHECK(evolved_2.GetCellState(i, j) == State::Infect);
            }
        }
    }
    SUBCASE("Testing contagion near the edges")
    {
        test(0, 0) = State::Infect;

        Board evolved_1 = EvolveTestContagion(test);
        CHECK(evolved_1.GetCellState(0, 1) == State::Infect);
        CHECK(evolved_1.GetCellState(1, 0) == State::Infect);
        CHECK(evolved_1.GetCellState(1, 1) == State::Infect);
        for (int j = 2; j != N; j++)
        {
            CHECK(evolved_1.GetCellState(0, j) == State::Susceptible);
            CHECK(evolved_1.GetCellState(1, j) == State::Susceptible);
        }
        for (int j = 0; j != N; j++)
        {
            CHECK(evolved_1.GetCellState(2, j) == State::Susceptible);
        }

        Board evolved_2 = EvolveTestContagion(evolved_1);
        for (int i = 0; i != 3; i++)
        {
            for (int j = 0; j != 3; j++)
            {
                CHECK(evolved_2.GetCellState(i, j) == State::Infect);
            }
            for (int j = 3; j != N; j++)
            {
                CHECK(evolved_2.GetCellState(i, j) == State::Susceptible);
            }
        }
        for (int i = 3; i != N; i++)
        {
            for (int j = 0; j != N; j++)
            {
                CHECK(evolved_2.GetCellState(i, j) == State::Susceptible);
            }
        }
    }
    SUBCASE("Testing Healing")
    {
        test(0, 0) = State::Infect;
        test(2, 2) = State::Infect;
        test(4, 4) = State::Infect;
        test(7, 8) = State::Infect;

        Board evolved_1 = EvolveTestHeal(test);
        CHECK(evolved_1.GetCellState(0, 0) == State::Recovered);
        CHECK(evolved_1.GetCellState(2, 2) == State::Recovered);
        CHECK(evolved_1.GetCellState(4, 4) == State::Recovered);
        CHECK(evolved_1.GetCellState(7, 8) == State::Recovered);
    }
}