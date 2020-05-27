#include <thread>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include "board.hpp"
#include "display.hpp"

int main()
{
    int const N = 175;
    Board test(N);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> infected(0., N);
    for (int i = 0; i != (N * N) / 200; ++i)
    {
        test(infected(gen), infected(gen)) = State::Infect;
    }
    std::ofstream ofs;
    std::ofstream ofs2;
    std::ofstream ofs3;
    std::ofstream ofs4;
    ofs.open("/home/luca/root/programmazione/exam_option_2/dati_S.dat", std::ifstream::out);
    ofs2.open("/home/luca/root/programmazione/exam_option_2/dati_I.dat", std::ifstream::out);
    ofs3.open("/home/luca/root/programmazione/exam_option_2/dati_R.dat", std::ifstream::out);
    ofs4.open("/home/luca/root/programmazione/exam_option_2/dati_D.dat", std::ifstream::out);

    /*
    if (ofs.is_open() && ofs2.is_open() && ofs3.is_open() && ofs4.is_open())
    {
        for (int i = 0; i != 100; ++i)
        {
            ofs << i << " " << test.GetSIRD().S << '\n';
            ofs2 << i << " " << test.GetSIRD().I << '\n';
            ofs3 << i << " " << test.GetSIRD().R << '\n';
            ofs4 << i << " " << test.GetSird().D << '\n';
            test.print();
            test = evolve(test);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        ofs.close();
        ofs2.close();
        ofs3.close();
        ofs4.close();
    }
    else
    {
        std::cout << "Couldn't open the files" << '\n';
    }
    */
    Display display{N};
    display.draw(test);
    std::cout << "Press any key to start the simulation. \n";
    display.WaitKeyPressed();
    if (!ofs.is_open() || !ofs2.is_open() || !ofs3.is_open() || !ofs4.is_open())
    {
        std::cout << "Couldn't open the files" << '\n';
    }
    if (display.WaitKeyPressed() == true && ofs.is_open() && ofs2.is_open() && ofs3.is_open() && ofs4.is_open())
    {
        for (int i = 0; i != 80; ++i)
        {
            ofs << i << " " << test.GetSIRD().S << '\n';
            ofs2 << i << " " << test.GetSIRD().I << '\n';
            ofs3 << i << " " << test.GetSIRD().R << '\n';
            ofs4 << i << " " << test.GetSIRD().D << '\n';
            test = evolve(test);
            display.draw(test);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ofs.close();
        ofs2.close();
        ofs3.close();
        ofs4.close();
        std::cout << "Press any key to close the window. \n";
        display.WaitKeyPressed();
        std::cout << "Program closed" << '\n';
    }
    if (display.WaitKeyPressed() == false)
    {
        std::cout << "Program closed" << '\n';
    }
}
