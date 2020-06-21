#include <memory>
#include <iostream>
#include "board.hpp"

std::ostream &operator<<(std::ostream &os, State const &s)
{
    switch (s)
    {
    case State::Susceptible:
        os << "Susceptible";
        break;
    case State::Infect:
        os << "Infected";
        break;
    case State::Recovered:
        os << "Recovered";
        break;
    case State::Dead:
        os << "Dead";
        break;
    }
    return os;
}

int main()
{
    std::unique_ptr<Board> up = std::make_unique<Board>(5);
    auto up2 = std::move(up); // ok, unique_ptr can be moved
    //up->print(); up cannot be used anymore after having been moved 
    //auto up2 = up; unique_ptr cannot be copied 
    up2->operator()(1,1) = State::Infect;
    up2->operator()(2,2) = State::Infect;
    up2->operator()(4,4) = State::Dead;
    up2->print();
    for(int i = 0; i != up2->size(); i++)
    {
        for(int j = 0; j != up2->size(); j++)
        {
            std::cout << "Cell (" << i << ',' << j << ')' << " is " << up2->GetCellState(i, j) << '\n';
        }
    }
}