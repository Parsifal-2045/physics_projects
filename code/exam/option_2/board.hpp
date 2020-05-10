#ifndef BOARD_HPP
#define BOARD_HPP

#include <vector>
#include <cassert>
#include <iostream>
#include <SFML/Graphics.hpp>

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

    auto size() const { return size_; }

    auto data() const { return data_; }

    State &operator()(int x, int y)
    {
        assert(x >= 0 && x < size_ && y >= 0 && y < size_);
        return data_[x * size_ + y];
    }

    State GetCellState(int x, int y) const
    {
        if (x > size_ || y > size_ || x < 0 || y < 0)
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
        return infect;
    }

    void print()
    {
        std::cout << "\033c";
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
                    std::cout << "i ";
                }
                if (status == 2)
                {
                    std::cout << "x ";
                }
            }
            std::cout << '\n';
        }
    }
};

#endif