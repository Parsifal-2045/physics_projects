#include "resonance_type.hpp"

#include <iostream>

ResonanceType::ResonanceType(std::string const &name, double const mass, int const charge,
                             double const width) : ParticleType(name, mass, charge), width_{width} {}

ResonanceType::~ResonanceType() = default;

double ResonanceType::GetWidth() const
{
    return width_;
}

void ResonanceType::Print() const
{
    ParticleType::Print();
    std::cout << "Width: " << width_ << '\n';
}