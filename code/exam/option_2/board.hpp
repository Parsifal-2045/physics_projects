#ifndef BOARD_HPP
#define BOARD_HPP

#include <vector>
#include <cassert>
#include <iostream>
#include <random>

enum class State : char
{
    Susceptible,
    Infect,
    Recovered,
    Dead
};

struct SIRD
{
    int S = 0;
    int I = 0;
    int R = 0;
    int D = 0;
};

class Board
{
private:
    int size_;
    std::vector<State> data_;

public:
    Board(int n) : size_{n}, data_(n * n) {}

    int size() const
    {
        return size_;
    }

    auto data() const
    {
        return data_;
    }

    State &operator()(int i, int j)
    {
        assert(i >= 0 && i < size_ && j >= 0 && j < size_);
        return data_[i * size_ + j];
    }

    State GetCellState(int i, int j) const
    {
        if (i >= size_ || j >= size_ || i < 0 || j < 0)
        {
            return State::Recovered;
        }
        else
        {
            return data_[i * size_ + j];
        }
    }

    auto CheckNeighbours(int i, int j) const
    {
        int infect = 0;
        if (GetCellState(i, j) == State::Infect)
        {
            --infect;
        }
        if (j == 0)
        {
            for (int row = -1; row != 2; row++)
            {
                for (int column = 0; column != 2; column++)
                {
                    if (GetCellState(i + row, j + column) == State::Infect)
                    {
                        ++infect;
                    }
                }
            }
        }
        if (j == size_ - 1)
        {
            for (int row = -1; row != 2; row++)
            {
                for (int column = -1; column != 1; column++)
                {
                    if (GetCellState(i + row, j + column) == State::Infect)
                    {
                        ++infect;
                    }
                }
            }
        }
        else
        {
            for (int row = -1; row != 2; row++)
            {
                for (int column = -1; column != 2; column++)
                {
                    if (GetCellState(i + row, j + column) == State::Infect)
                    {
                        ++infect;
                    }
                }
            }
        }
        assert(infect >= 0 && infect <= 8);
        return infect;
    }

    SIRD GetSIRD()
    {
        int sus = 0;
        int inf = 0;
        int rec = 0;
        int dead = 0;
        for (int i = 0; i != size_; i++)
        {
            for (int j = 0; j != size_; j++)
            {
                if (GetCellState(i, j) == State::Susceptible)
                {
                    ++sus;
                }
                if (GetCellState(i, j) == State::Infect)
                {
                    ++inf;
                }
                if (GetCellState(i, j) == State::Recovered)
                {
                    ++rec;
                }
                if (GetCellState(i, j) == State::Dead)
                {
                    ++dead;
                }
            }
        }
        assert(sus + inf + rec + dead == static_cast<int>(data_.size()));
        return SIRD{sus, inf, rec, dead};
    }

    void print()
    {
        std::cout << "\033c";
        const std::string red("\033[0;31m");
        const std::string green("\033[1;32m");
        const std::string yellow("\033[1;33m");
        const std::string cyan("\033[0;36m");
        const std::string magenta("\033[0;35m");
        const std::string reset("\033[0m");
        for (int i = 0; i != size_; i++)
        {
            for (int j = 0; j != size_; j++)
            {
                auto status = static_cast<int>(data_[i * size_ + j]);
                assert(status >= 0 && status <= 3);
                if (status == 0)
                {
                    std::cout << "o ";
                }
                if (status == 1)
                {
                    std::cout << magenta << "i " << reset;
                }
                if (status == 2)
                {
                    std::cout << green << "o " << reset;
                }
                if (status == 3)
                {
                    std::cout << red << "x " << reset;
                }
            }
            std::cout << '\n';
        }
        std::cout << "Susceptibles : " << GetSIRD().S << " | "
                  << "Infected : " << GetSIRD().I << " | "
                  << "Recovered : " << GetSIRD().R << " | "
                  << "Dead : " << GetSIRD().D << '\n';
    }
};

inline bool operator==(Board const &lhs, Board const &rhs)
{
    return lhs.data() == rhs.data();
}

inline Board evolve(Board const &current)
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0., 1.);
    auto contagion_probability = dis(gen);
    auto heal_probability = dis(gen);
    auto death_probability = dis(gen);
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
            if (current.GetCellState(i, j) == State::Dead)
            {
                next(i, j) = current.GetCellState(i, j);
                assert(next.GetCellState(i, j) == State::Dead);
            }
        }
    }
    return next;
}

#endif