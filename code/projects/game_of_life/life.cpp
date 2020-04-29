#include <vector>
#include <iostream>
#include <cassert>

enum class State : char
{
    Dead,
    Alive
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

    State operator()(int const &row, int const &column) const //Returns state of the cell in position (row,column)
    {
        if (row > size_ || column > size_)
        {
            return State::Dead;
        }
        else
        {
            int r = row * size_ + column;
            return data_[r];
        }
    }

    State &operator()(int const &row, int const &column)
    {
        assert(row >= 0 && row < size_ && column >= 0 && column < size_);
        int r = row * size_ + column;
        return data_[r];
    }

    int GetNeighbours(int const &row, int const &column) const //Returns the number of living neighbours of a cell
    {
        int pos = row * size_ + column;
        int up = pos - row;
        int down = pos + row;
        int alive = 0;
        for (int i = 0; i != 3; i++)
        {
            if (data_[up + i] == State::Alive)
            {
                ++alive;
            }
        }
        for (int i = 0; i != 3; i++)
        {
            if (data_[down + i] == State::Alive)
            {
                ++alive;
            }
        }
        if (data_[pos - 1] == State::Alive)
        {
            ++alive;
        }
        if (data_[pos + 1] == State::Alive)
        {
            ++alive;
        }
        return alive;
    }

    void print()
    {
        auto it = data_.begin();
        while (it < data_.end())
        {
            for (int i = 0; i != size_; ++i, ++it)
            {
                if (*it == State::Alive)
                {
                    std::cout << "1 ";
                }
                else
                {
                    std::cout << "0 ";
                }
            }
            std::cout << '\n';
            ++it;
        }
    }
};

Board evolve(Board const &current)
{
    int const N = current.size();
    Board next(N);
    for (int i = 0; i != N; ++i)
    {
        for (int j = 0; j != N; ++j)
        {
            int const alive = current.GetNeighbours(i, j);
            assert(alive >= 0 && alive <= 8);
            if (alive == 3)
            {
                next(i, j) = State::Alive;
            }
            if (alive == 2)
            {
                next(i, j) = current(i, j);
            }
            if (alive < 2 || alive > 3)
            {
                next(i, j) = State::Dead;
            }
        }
    }
    return next;
}

int main()
{
    // Inizializzazione

    constexpr int dim = 10;
    Board board(dim);
    board(0, 0) = State::Alive;
    board(0, 1) = State::Alive;
    board(0, 2) = State::Alive;

    board.print();
    std::cout << '\n';

    //Evoluzione

    for (int c = 0; c != 150; c++)
    {
        board = evolve(board);
    }
    
    // Display

    board.print();
}