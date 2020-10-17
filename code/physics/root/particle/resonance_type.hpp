#ifndef RESONANCE_TYPE_HPP
#define RESONANCE_TYPE_HPP

#include "particle_type.hpp"

class ResonanceType : public ParticleType
{
private:
    std::string const name_;
    double const mass_;
    int const charge_;
    double const width_;

public:
    ResonanceType(std::string const &name, double const mass, int const charge, double const width);
    double const GetWidth();
    void print() const;
};

#endif