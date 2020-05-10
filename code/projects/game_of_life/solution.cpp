#include <cassert>
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>

enum class State : char
{
    Dead,
    Alive
};

class Board
{
    int n_;
    std::vector<State> board_;

public:
    Board(int n) : n_(n), board_(n * n) {}

    State operator()(int i, int j) const
    {
        return (i >= 0 && i < n_ && j >= 0 && j < n_) ? board_[i * n_ + j] : State::Dead;
    }
    State &operator()(int i, int j)
    {
        assert(i >= 0 && i < n_ && j >= 0 && j < n_);
        return board_[i * n_ + j];
    }
    int size() const
    {
        return n_;
    }
    void print() const
    {
        std::cout << "\033c";
        for (int i = 0; i != n_; i++)
        {
            for (int j = 1; j != n_; j++)
            {
                char status = (static_cast<int>(board_[i * n_ + j])) ? '+' : ' ';
                std::cout << status << " ";
            }
            std::cout << '\n';
        }
    }
};
int neigh_count(Board const &board, int const r, int const c)
{
    int result = -static_cast<int>(board(r, c));
    for (int i = r - 1; i != r + 2; ++i)
    {
        for (int j = c - 1; j != c + 2; ++j)
        {
            result += static_cast<int>(board(i, j));
        }
    }
    return result;
}

Board evolve(Board const &current)
{
    int const N = current.size();
    Board next(N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int const c = neigh_count(current, i, j);
            if (c == 3)
            {
                next(i, j) = State::Alive;
            }
            else if (c == 2)
            {
                next(i, j) = current(i, j);
            }
        }
    }

    return next;
}

int main()
{
    int dim = 50;
    Board board(dim);

    board(25, 24) = State::Alive;
    board(26, 25) = State::Alive;
    board(26, 23) = State::Alive;
    board(26, 24) = State::Alive;
    board(26, 25) = State::Alive;

    for (int i = 0; i != 150; ++i)
    {
        board = evolve(board);
        board.print();
        std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    }
}
