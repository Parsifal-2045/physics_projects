#include <iostream>
#include <random>
#include <cassert>
#include <fstream>

struct SIR
{
    int S;
    int I;
    int R;
    int N;
};

SIR evolve(SIR const &current)
{
    /*
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0., 1.);
    auto beta = dis(gen);
    auto gamma = dis(gen);
    assert(beta >= 0 && beta <= 1 && gamma >= 0 && gamma <= 1);
    */
    double beta = 0.8;
    double gamma = 0.4;

    SIR next{};

    next.S = current.S - ((beta / current.N) * current.I * current.S);
    next.I = current.I + ((beta / current.N) * current.I * current.S) - (gamma * current.I);
    next.R = current.R + (gamma * current.I);

    if (next.S < 0)
    {
        next.S = 0;
    }
    if (next.R < 0)
    {
        next.R = 0;
    }
    if (next.I < 0)
    {
        next.I = 0;
    }

    assert(next.N = next.S + next.I + next.R);

    return next;
}

void print(SIR s, int const &n)
{
    for (int i = 0; i != n; i++)
    {
        std::cout << i << " " << s.S << " " << s.I << " " << s.R << '\n';
        s = evolve(s);
    }
}

void printfile(SIR s, int const &n)
{
    std::ofstream ofs;
    std::ofstream ofs2;
    std::ofstream ofs3;

    //ofs.open("dati_S.dat", std::ifstream::out);
    //ofs2.open("dati_I.dat", std::ifstream::out);
    //ofs3.open("dati_R.dat", std::ifstream::out);

    ofs.open("/home/luca/root/covid19/dati_S.dat", std::ifstream::out);
    ofs2.open("/home/luca/root/covid19/dati_I.dat", std::ifstream::out);
    ofs3.open("/home/luca/root/covid19/dati_R.dat", std::ifstream::out);
    if (ofs.is_open() && ofs2.is_open() && ofs3.is_open())
    {
        for (int i = 0; i != n; i++)
        {
            ofs << i << " " << s.S << '\n';
            ofs2 << i << " " << s.I << '\n';
            ofs3 << i << " " << s.R << '\n';
            s = evolve(s);
        }
        ofs.close();
        ofs2.close();
        ofs3.close();
    }
    else
    {
        std::cout << "Couldn't open the file" << '\n';
    }
}

int main()
{
    int S = 1000;
    int I = 500;
    int R = 0;
    int N = S + I + R;
    SIR test{S, I, R, N};
    //print(test, 100);
    printfile(test, 100);
}