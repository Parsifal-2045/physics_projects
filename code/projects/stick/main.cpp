#include <iostream>
#include <thread>
#include <chrono>
#include "board.hpp"
#include "display.hpp"

auto stick(Board &test)
{
    // S
    test(7, 11) = State::Infect;
    test(7, 12) = State::Infect;
    test(7, 13) = State::Infect;
    test(7, 14) = State::Infect;
    test(8, 11) = State::Infect;
    test(9, 11) = State::Infect;
    test(9, 12) = State::Infect;
    test(9, 13) = State::Infect;
    test(9, 14) = State::Infect;
    test(10, 14) = State::Infect;
    test(11, 14) = State::Infect;
    test(11, 13) = State::Infect;
    test(11, 12) = State::Infect;
    test(11, 11) = State::Infect;
    //T
    test(7, 16) = State::Infect;
    test(7, 17) = State::Infect;
    test(7, 18) = State::Infect;
    test(7, 19) = State::Infect;
    test(7, 20) = State::Infect;
    test(8, 18) = State::Infect;
    test(9, 18) = State::Infect;
    test(10, 18) = State::Infect;
    test(11, 18) = State::Infect;
    //I
    test(7, 22) = State::Infect;
    test(8, 22) = State::Infect;
    test(9, 22) = State::Infect;
    test(10, 22) = State::Infect;
    test(11, 22) = State::Infect;
    //C
    test(7, 24) = State::Infect;
    test(7, 25) = State::Infect;
    test(7, 26) = State::Infect;
    test(7, 27) = State::Infect;
    test(8, 24) = State::Infect;
    test(9, 24) = State::Infect;
    test(10, 24) = State::Infect;
    test(11, 24) = State::Infect;
    test(11, 25) = State::Infect;
    test(11, 26) = State::Infect;
    test(11, 27) = State::Infect;
    //K
    test(7, 29) = State::Infect;
    test(8, 29) = State::Infect;
    test(9, 29) = State::Infect;
    test(10, 29) = State::Infect;
    test(11, 29) = State::Infect;
    test(9, 30) = State::Infect;
    test(8, 31) = State::Infect;
    test(7, 32) = State::Infect;
    test(10, 31) = State::Infect;
    test(11, 32) = State::Infect;

    //A
    test(13, 5) = State::Infect;
    test(13, 6) = State::Infect;
    test(13, 7) = State::Infect;
    test(13, 8) = State::Infect;
    test(13, 9) = State::Infect;
    test(14, 5) = State::Infect;
    test(15, 5) = State::Infect;
    test(16, 5) = State::Infect;
    test(17, 5) = State::Infect;
    test(15, 6) = State::Infect;
    test(15, 7) = State::Infect;
    test(15, 8) = State::Infect;
    test(15, 9) = State::Infect;
    test(14, 9) = State::Infect;
    test(15, 9) = State::Infect;
    test(16, 9) = State::Infect;
    test(17, 9) = State::Infect;
    //G
    test(13, 11) = State::Infect;
    test(13, 12) = State::Infect;
    test(13, 13) = State::Infect;
    test(13, 14) = State::Infect;
    test(14, 11) = State::Infect;
    test(15, 11) = State::Infect;
    test(16, 11) = State::Infect;
    test(17, 11) = State::Infect;
    test(17, 12) = State::Infect;
    test(17, 13) = State::Infect;
    test(17, 14) = State::Infect;
    test(16, 14) = State::Infect;
    test(15, 13) = State::Infect;
    test(15, 14) = State::Infect;
    //A
    test(13, 16) = State::Infect;
    test(13, 17) = State::Infect;
    test(13, 18) = State::Infect;
    test(13, 19) = State::Infect;
    test(13, 20) = State::Infect;
    test(14, 16) = State::Infect;
    test(15, 16) = State::Infect;
    test(16, 16) = State::Infect;
    test(17, 16) = State::Infect;
    test(14, 20) = State::Infect;
    test(15, 20) = State::Infect;
    test(16, 20) = State::Infect;
    test(17, 20) = State::Infect;
    test(15, 17) = State::Infect;
    test(15, 18) = State::Infect;
    test(15, 19) = State::Infect;
    test(15, 20) = State::Infect;
    //I
    test(13, 22) = State::Infect;
    test(14, 22) = State::Infect;
    test(15, 22) = State::Infect;
    test(16, 22) = State::Infect;
    test(17, 22) = State::Infect;
    //N
    test(13, 24) = State::Infect;
    test(14, 24) = State::Infect;
    test(15, 24) = State::Infect;
    test(16, 24) = State::Infect;
    test(17, 24) = State::Infect;
    test(13, 27) = State::Infect;
    test(14, 27) = State::Infect;
    test(15, 27) = State::Infect;
    test(16, 27) = State::Infect;
    test(17, 27) = State::Infect;
    test(14, 25) = State::Infect;
    test(15, 26) = State::Infect;
    //S
    for (int i = 0; i != 4; i++)
    {
        test(13, 29 + i) = State::Infect;
    }
    for (int i = 0; i != 3; i++)
    {
        test(13 + i, 29) = State::Infect;
    }
    for (int i = 0; i != 4; i++)
    {
        test(15, 29 + i) = State::Infect;
    }
    for (int i = 0; i != 3; i++)
    {
        test(15 + i, 32) = State::Infect;
    }
    for (int i = 0; i != 4; i++)
    {
        test(17, 29 + i) = State::Infect;
    }
    //T
    for (int i = 0; i != 5; i++)
    {
        test(13, 34 + i) = State::Infect;
    }
    for (int i = 0; i != 4; i++)
    {
        test(14 + i, 36) = State::Infect;
    }

    //Y
    test(19, 12) = State::Infect;
    for (int i = 0; i != 3; i++)
    {
        test(20, 12 + i) = State::Infect;
    }
    test(19, 14) = State::Infect;
    for (int i = 0; i != 4; i++)
    {
        test(20 + i, 13) = State::Infect;
    }
    //O
    for (int i = 0; i != 5; i++)
    {
        test(19, 16 + i) = State::Infect;
        test(19 + i, 16) = State::Infect;
        test(19 + i, 20) = State::Infect;
        test(23, 16 + i) = State::Infect;
    }
    //U
    for (int i = 0; i != 5; i++)
    {
        test(19 + i, 22) = State::Infect;
        test(23, 22 + i) = State::Infect;
        test(19 + i, 26) = State::Infect;
    }
    //R
    for (int i = 0; i != 4; i++)
    {
        test(19, 28 + i) = State::Infect;
        test(20 + i, 28) = State::Infect;
        test(21, 28 + i) = State::Infect;
    }
    test(20, 31) = State::Infect;
    test(22, 30) = State::Infect;
    test(23, 31) = State::Infect;

    //N
    for (int i = 0; i != 5; i++)
    {
        test(25 + i, 12) = State::Infect;
        test(25 + i, 15) = State::Infect;
    }
    test(26, 13) = State::Infect;
    test(27, 14) = State::Infect;
    //E
    for (int i = 0; i != 4; i++)
    {
        test(25, 17 + i) = State::Infect;
        test(26 + i, 17) = State::Infect;
        test(29, 17 + i) = State::Infect;
    }
    test(27, 18) = State::Infect;
    test(27, 19) = State::Infect;
    //C
    for (int i = 0; i != 4; i++)
    {
        test(25, 22 + i) = State::Infect;
        test(26 + i, 22) = State::Infect;
        test(29, 22 + i) = State::Infect;
    }
    //K
    for (int i = 0; i != 5; i++)
    {
        test(25 + i, 27) = State::Infect;
    }
    test(27, 28) = State::Infect;
    test(26, 29) = State::Infect;
    test(25, 30) = State::Infect;
    test(28, 29) = State::Infect;
    test(29, 30) = State::Infect;

    return test;
}

int main()
{
    int const N = 44;
    Board b(N);
    b = stick(b);
    b.print();
    std::cout << "Press enter to start" << '\n';
    std::cin.get();
    for (int i = 0; i != 30; i++)
    {
        b = evolve(b);
        b.print();
        if (b.GetSIRD().I == 0)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "Evolution completed" << '\n';
    /*
    Display display{N};
    display.draw(b);
    display.WaitKeyPressed();
    for (int i = 0; i != 30; i++)
    {
        Board temp = b;
        b = evolve(b);
        display.draw(b);
        if (temp.GetSIRD().I == 0 && b.GetSIRD().I == 0)
        {
            std::cout << "Evolution completed" << '\n';
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    display.WaitKeyPressed();
    */
}