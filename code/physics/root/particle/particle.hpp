#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include "particle_type.hpp"
#include "resonance_type.hpp"

class Particle
{
private:
    static int const MaxNumParticleType_ = 9;
    static std::vector<ParticleType *> Index_;
    static int NParticleType_;
    int IndexParticle_;
    double Px_;
    double Py_;
    double Pz_;
    static int FindParticle(std::string const &name);
    void Boost(double bx, double by, double bz);

public:
    Particle(std::string const &name, double Px, double Py, double Pz);
    static void Destructor();
    int GetIndexPosition() const;
    double GetMass() const;
    double GetPx() const;
    double GetPy() const;
    double GetPz() const;
    static void AddParticleType(std::string const &name, double const mass, int const charge, double width);
    void SetAttribute(int const i);
    void SetAttribute(std::string const &name);
    double GetEnergy() const;
    double InvMass(Particle &p) const;
    void SetP(double Px, double Py, double Pz);
    int Decay2Body(Particle &dau1, Particle &dau2) const;
    static void PrintIndex();
    void PrintParticle();
};

#endif