#ifndef PARTICLE_TYPE_HPP
#define PARTICLE_TYPE_HPP

#include <string>

class ParticleType
{
private:
    std::string const name_;
    double const mass_;
    int const charge_;

public:
    ParticleType(std::string const &name, double const mass, int const charge);
    std::string const GetName();
    double const GetMass();
    int const GetCharge();
    virtual void print() const;
};

#endif