#include "resonance_type.hpp"

#include <iostream>

ResonanceType::ResonanceType(std::string const &name = {}, double const mass = {}, int const charge = {},
                             double const width = {}) : name_{name}, mass_{mass}, charge_{charge},
                                                        width_{width} {}

double const ResonanceType::GetWidth()
{
    return width_;
}

void ResonanceType::print() const
{
    ParticleType::print();
    std::cout << "Resonance width: " << width_ << '\n';
}