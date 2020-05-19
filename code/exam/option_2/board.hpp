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
    Recovered
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

    State &operator()(int x, int y)
    {
        assert(x >= 0 && x < size_ && y >= 0 && y < size_);
        return data_[x * size_ + y];
    }

    State GetCellState(int x, int y) const
    {
        if (x >= size_ || y >= size_ || x < 0 || y < 0)
        {
            return State::Recovered;
        }
        else
        {
            return data_[x * size_ + y];
        }
    }

    auto CheckNeighbours(int x, int y) const // Checks if there are infected cells nearby
    {
        int infect = 0;
        if (y == 0)
        {
            //upper left border cells

            for (int i = 0; i != 2; i++)
            {
                if (GetCellState(x - 1, y + i) == State::Infect)
                {
                    ++infect;
                }
            }

            //middle left border cells

            if (GetCellState(x, y + 1) == State::Infect)
            {
                ++infect;
            }

            //lower left border cells

            for (int i = 0; i != 2; i++)
            {
                if (GetCellState(x + 1, y + i) == State::Infect)
                {
                    ++infect;
                }
            }
        }

        if (y == size_ - 1)
        {
            //upper right border cells

            for (int i = -1; i != 1; i++)
            {
                if (GetCellState(x + 1, y + i) == State::Infect)
                {
                    ++infect;
                }
            }

            //middle right border cells

            if (GetCellState(x, y - 1) == State::Infect)
            {
                ++infect;
            }

            //lower right border cells

            for (int i = 0; i != 1; i++)
            {
                if (GetCellState(x + 1, y + i) == State::Infect)
                {
                    ++infect;
                }
            }
        }

        else
        {
            // upper cells

            for (int i = -1; i != 2; i++)
            {
                if (GetCellState(x - 1, y + i) == State::Infect)
                {
                    ++infect;
                }
            }

            // middle cells

            if (GetCellState(x, y - 1) == State::Infect)
            {
                ++infect;
            }

            if (GetCellState(x, y + 1) == State::Infect)
            {
                ++infect;
            }

            // lower cells

            for (int i = -1; i != 2; i++)
            {
                if (GetCellState(x + 1, y + i) == State::Infect)
                {
                    ++infect;
                }
            }
        }
        return infect;
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
                assert(status >= 0 && status <= 2);
                if (status == 0)
                {
                    std::cout << "o ";
                }
                if (status == 1)
                {
                    std::cout << red << "i " << reset;
                }
                if (status == 2)
                {
                    std::cout << green << "x " << reset;
                }
            }
            std::cout << '\n';
        }
    }
};

inline Board evolve(Board const &current)
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

#endif