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
            if (current.GetCellState(i, j) == State::Dead)
            {
                next(i, j) = current.GetCellState(i, j);
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
            if (current.GetCellState(i, j) == State::Dead)
            {
                next(i, j) = current.GetCellState(i, j);
            }
        }
    }
    return next;
}

Board EvolveTestDeath(Board const &current)
{
    auto contagion_probability = 0;
    auto heal_probability = 0;
    auto death_probability = 1;
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
                if (heal_probability >= status)
                {
                    next(i, j) = State::Recovered;
                }
                else if (death_probability >= status)
                {
                    next(i, j) = State::Dead;
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
            if (current.GetCellState(i, j) == State::Dead)
            {
                next(i, j) = current.GetCellState(i, j);
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
    CHECK(test.GetCellState(N + 1, N + 1) == State::Recovered);
    CHECK(test.GetCellState(-1, -1) == State::Recovered);

    SUBCASE("Testing infection rate in the board")
    {
        test(2, 3) = State::Infect;

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

        Board evolved_2 = EvolveTestContagion(evolved_1);
        for (int i = 0; i != 5; i++)
        {
            for (int j = 1; j != 6; j++)
            {
                CHECK(evolved_2.GetCellState(i, j) == State::Infect);
            }
        }
    }

    SUBCASE("Testing GetSIRD")
    {
        auto initial_state = test.GetSIRD();
        CHECK(initial_state.S == N * N);
        CHECK(initial_state.I == 0);
        CHECK(initial_state.R == 0);
        CHECK(initial_state.D == 0);

        test(2, 2) = State::Infect;
        CHECK(test.GetSIRD().S == (N * N) - 1);
        CHECK(test.GetSIRD().I == 1);
        CHECK(test.GetSIRD().R == 0);
        CHECK(test.GetSIRD().D == 0);
        CHECK(test.GetSIRD().S + test.GetSIRD().I + test.GetSIRD().R + test.GetSIRD().D == N * N);

        Board evolved = EvolveTestContagion(test);
        CHECK(evolved.GetSIRD().S == (N * N) - 9);
        CHECK(evolved.GetSIRD().I == 9);
        CHECK(evolved.GetSIRD().R == 0);
        CHECK(evolved.GetSIRD().D == 0);
        CHECK(evolved.GetSIRD().S + evolved.GetSIRD().I + evolved.GetSIRD().R + evolved.GetSIRD().D == N * N);

        Board evolved2 = EvolveTestContagion(evolved);
        CHECK(evolved2.GetSIRD().S == (N * N) - 25);
        CHECK(evolved2.GetSIRD().I == 25);
        CHECK(evolved2.GetSIRD().R == 0);
        CHECK(evolved2.GetSIRD().D == 0);
        CHECK(evolved2.GetSIRD().S + evolved2.GetSIRD().I + evolved2.GetSIRD().R + evolved2.GetSIRD().D == N * N);

        Board healed = EvolveTestHeal(evolved2);
        CHECK(healed.GetSIRD().S == (N * N) - 25);
        CHECK(healed.GetSIRD().I == 0);
        CHECK(healed.GetSIRD().R == 25);
        CHECK(healed.GetSIRD().D == 0);
        CHECK(healed.GetSIRD().S + healed.GetSIRD().I + healed.GetSIRD().R + healed.GetSIRD().D == N * N);
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
    SUBCASE("Testing Death")
    {
        test(1, 1) = State::Dead;

        CHECK(test.GetCellState(1, 1) == State::Dead);
        for (int i = 0; i != 100; i++)
        {
            Board b = EvolveTestContagion(test);
            CHECK(b.GetCellState(1, 1) == State::Dead);
            CHECK(b.GetSIRD().S == N * N - 1);
            CHECK(b.GetSIRD().I == 0);
            CHECK(b.GetSIRD().R == 0);
            CHECK(b.GetSIRD().D == 1);
            CHECK(b.GetSIRD().S + b.GetSIRD().I + b.GetSIRD().R + b.GetSIRD().D == N * N);
        }
    }
}