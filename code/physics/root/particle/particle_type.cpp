#include "particle_type.hpp"
#include <iostream>

ParticleType::ParticleType(std::string const &name, double const mass, int const charge)
    : name_{name}, mass_{mass}, charge_{charge} {}

ParticleType::~ParticleType() = default;

std::string ParticleType::GetName() const
{
    return name_;
}

double ParticleType::GetMass() const
{
    return mass_;
}

int ParticleType::GetCharge() const
{
    return charge_;
}

void ParticleType::Print() const
{
    std::cout << "Name: " << name_ << '\n';
    std::cout << "Mass: " << mass_ << '\n';
    std::cout << "Charge: " << charge_ << '\n';
}