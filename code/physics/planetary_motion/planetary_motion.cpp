#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>

constexpr double G = 6.674e-11;
constexpr double M_S = 1.989e30;
constexpr double k = G * M_S;
constexpr double cf_au = 1.496e11;

struct State
{
    double x;
    double y;
    double vx;
    double vy;
    double ax;
    double ay;
};

enum class Unit {au, kilometer, meter};

class Orbit
{
private:
    State m_s0;
    double m_rp;

public:
    Orbit(State s0, double rp) : m_s0{s0}, m_rp{rp} {}

    auto createOrbit(int const &N_point)
    {
        std::vector<State> result;
        result.reserve(N_point);
        State previous = m_s0;
        result.push_back(previous);
        double const epsilon = m_rp / N_point;
        for (int i = 1; i != N_point; ++i)
        {
            State s;
            s.ax = -k * previous.x / pow(previous.x * previous.x + previous.y * previous.y, 1.5);
            s.ay = -k * previous.y / pow(previous.x * previous.x + previous.y * previous.y, 1.5);
            s.vx = previous.vx + epsilon * s.ax;
            s.vy = previous.vy + epsilon * s.ay;
            s.x = previous.x + epsilon * s.vx;
            s.y = previous.y + epsilon * s.vy;
            result.push_back(s);
            previous = s;
        }
        return result;
    }
};

double convert(double const &value, Unit unit)
{
    switch (unit)
    {
    case Unit::meter:
        return value;
    case Unit::kilometer:
        return value / 1000;
    case Unit::au:
        return value / cf_au;
    };
}

void printOrbitCoordinates(std::vector<State> const &planet_orbit, Unit unit)
{
    for (auto it = planet_orbit.begin(), end = planet_orbit.end(); it != end; ++it)
    {
        auto v = *it;
        std::cout << convert(v.x, unit) << ',' << convert(v.y, unit) << '\n';
    }
}

int main()
{
    State s0{};
    s0.x = 1.470568e11; // perihelion of Earth in [m]
    s0.vy = 3.028361e4; // speed of revolution of the Earth in [m/s]
    // period of revolution 365.25 days, i.e. 3.15576e7 seconds
    Orbit earth(s0, 3.15576e7);
    // use 500 points to calculate the orbit
    auto earth_orbit = earth.createOrbit(500);
    printOrbitCoordinates(earth_orbit, Unit::au);
    //printOrbitCoordinates(earth_orbit, Unit::kilometer);
    //printOrbitCoordinates(earth_orbit, Unit::meter);
}