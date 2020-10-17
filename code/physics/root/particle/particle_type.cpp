#include "particle_type.hpp"
#include <iostream>

ParticleType::ParticleType(std::string const &name = {}, double const mass = {}, int const charge = {})
    : name_{name}, mass_{mass}, charge_{charge} {}

std::string const ParticleType::GetName()
{
    return name_;
}

double const ParticleType::GetMass()
{
    return mass_;
}

int const ParticleType::GetCharge()
{
    return charge_;
}

void ParticleType::print() const
{
    std::cout << "Particle name: " << name_ << '\n';
    std::cout << "Particle mass: " << mass_ << '\n';
    std::cout << "Particle charge: " << charge_ << '\n';
}