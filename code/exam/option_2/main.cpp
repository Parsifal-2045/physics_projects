#include <thread>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include "board.hpp"
#include "display.hpp"

int main()
{
    int const N = 45;
    Board test(N);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> infected(0., N);
    for (int i = 0; i != N / 10; ++i)
    {
        test(infected(gen), infected(gen)) = State::Infect;
    }
    std::ofstream ofs;
    std::ofstream ofs2;
    std::ofstream ofs3;
    ofs.open("/home/luca/root/programmazione/exam_option_2/dati_S.dat", std::ifstream::out);
    ofs2.open("/home/luca/root/programmazione/exam_option_2/dati_I.dat", std::ifstream::out);
    ofs3.open("/home/luca/root/programmazione/exam_option_2/dati_R.dat", std::ifstream::out);
    if (ofs.is_open() && ofs2.is_open() && ofs3.is_open())
    {
        for (int i = 0; i != 100; ++i)
        {
            ofs << i << " " << test.GetSIR().S << '\n';
            ofs2 << i << " " << test.GetSIR().I << '\n';
            ofs3 << i << " " << test.GetSIR().R << '\n'; 
            test.print();
            test = evolve(test);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        ofs.close();
        ofs2.close();
        ofs3.close();
    }
    else
    {
        std::cout << "Couldn't open the files" << '\n';
    }

    /*
    Display display{N};
    display.draw(test);
    std::cout << "Press any key to start the simulation. \n";
    display.WaitKeyPressed();
    for (int i = 0; i != 100; ++i)
    {
        test = evolve(test);
        display.draw(test);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    std::cout << "Press any key to close the window. \n";
    display.WaitKeyPressed();
    */
}
