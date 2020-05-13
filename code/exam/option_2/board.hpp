#ifndef BOARD_HPP
#define BOARD_HPP

#include <vector>
#include <cassert>
#include <iostream>

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

    int size() const;

    auto data() const;

    State &operator()(int x, int y);

    State GetCellState(int x, int y) const;

    auto CheckNeighbours(int x, int y) const; // Checks if there are infected cells nearby

    void print();
};

Board evolve(Board const &current);

#endif