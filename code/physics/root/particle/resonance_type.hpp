#ifndef RESONANCE_TYPE_HPP
#define RESONANCE_TYPE_HPP

#include "particle_type.hpp"

class ResonanceType : public ParticleType
{
private:
    double const width_;

public:
    ResonanceType(std::string const &name, double const mass, int const charge, double const width);
    ~ResonanceType();
    double GetWidth() const;
    void Print() const;
};

#endif