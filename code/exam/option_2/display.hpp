#ifndef BOARD_DISPLAY_HPP
#define BOARD_DISPLAY_HPP

#include "board.hpp"
#include <cassert>
#include <SFML/Graphics.hpp>

class Display
{
private:
    int board_side_;
    sf::RenderWindow window_;
    static constexpr int cell_size_ = 7;
    static int display_side(int board_side)
    {
        return (board_side - 6) * cell_size_;
    }
    static constexpr auto window_title = "Evolution of an epidemic";

public:
    Display(int board_side) : board_side_{board_side},
                              window_{sf::VideoMode(display_side(board_side), display_side(board_side)),
                                      window_title,
                                      sf::Style::Close}
    {
    }
    void draw(Board const &board)
    {
        assert(board_side_ == board.size());
        window_.clear(sf::Color::White);

        sf::RectangleShape infected(sf::Vector2f(cell_size_, cell_size_));
        infected.setFillColor(sf::Color::Red);
        infected.setOutlineThickness(1.f);
        infected.setOutlineColor(sf::Color::White);

        sf::RectangleShape recovered(sf::Vector2f(cell_size_, cell_size_));
        recovered.setFillColor(sf::Color::Green);
        recovered.setOutlineThickness(1.f);
        recovered.setOutlineColor(sf::Color::White);

        int const N = board.size();

        for (int i = 3; i != N - 3; ++i)
        {
            for (int j = 3; j != N - 3; ++j)
            {
                if (board.GetCellState(i, j) == State::Infect)
                {
                    infected.setPosition((j - 3) * cell_size_, (i - 3) * cell_size_);
                    window_.draw(infected);
                }
                if (board.GetCellState(i, j) == State::Recovered)
                {
                    recovered.setPosition((j - 3) * cell_size_, (i - 3) * cell_size_);
                    window_.draw(recovered);
                }
            }
        }
        window_.display();
    }
    void WaitKeyPressed()
    {
        sf::Event event;
        window_.waitEvent(event);
        while (event.type != sf::Event::KeyPressed)
        {
            window_.waitEvent(event);
        }
    }
};

#endif