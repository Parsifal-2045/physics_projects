#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "particle.hpp"

int Particle::NParticleType_ = 0;
std::vector<ParticleType *> Particle::Index_;

int Particle::FindParticle(std::string const &name)
{
    int result;
    bool found = false;
    int i = 0;
    for (auto value : Index_)
    {
        if (value->GetName() == name)
        {
            found = true;
            break;
        }
        else
        {
            i++;
        }
    }
    if (found)
    {
        result = i;
    }
    else
    {
        result = MaxNumParticleType_ + 1;
        //std::cout << "Particle not found" << '\n';
    }
    return result;
}

void Particle::Boost(double bx, double by, double bz)
{
    double energy = GetEnergy();
    double b2 = bx * bx + by * by + bz * bz;
    double gamma = 1.0 / std::sqrt(1.0 - b2);
    double bp = bx * Px_ + by * Py_ + bz * Pz_;
    double gamma2 = b2 > 0 ? (gamma - 1.0) / 2 : 0.0;
    Px_ += gamma2 * bp * bx + gamma * bx * energy;
    Py_ += gamma2 * bp * by + gamma * by * energy;
    Pz_ += gamma2 * bp * bz + gamma * bz * energy;
}

Particle::Particle(std::string const &name, double Px = 0, double Py = 0, double Pz = 0)
{
    Px_ = Px;
    Py_ = Py;
    Pz_ = Pz;
    IndexParticle_ = FindParticle(name);
}

void Particle::Destructor()
{
    for (auto value : Index_)
    {
        delete value;
    }
}

int Particle::GetIndexPosition() const
{
    return IndexParticle_;
}

double Particle::GetMass() const
{
    return Index_[IndexParticle_]->GetMass();
}

double Particle::GetPx() const
{
    return Px_;
}

double Particle::GetPy() const
{
    return Py_;
}

double Particle::GetPz() const
{
    return Pz_;
}

void Particle::AddParticleType(std::string const &name, double const mass, int const charge, double width = 0)
{
    if (Index_.size() <= MaxNumParticleType_)
    {
        if (FindParticle(name) == MaxNumParticleType_ + 1)
        {
            if (width == 0)
            {
                ParticleType *NewParticle = new ParticleType(name, mass, charge);
                Index_.push_back(NewParticle);
                NParticleType_ = Index_.size();
            }
            else
            {
                ResonanceType *NewResonance = new ResonanceType(name, mass, charge, width);
                Index_.push_back(NewResonance);
                NParticleType_ = Index_.size();
            }
        }
        else if (FindParticle(name) <= static_cast<int>(Index_.size()))
        {
            std::cout << "Already added " << name << '\n';
        }
    }
    else
    {
        std::cout << "Cannot add " << name << ". Maximum number of Particle Types reached" << '\n';
    }
}

void Particle::SetAttribute(int const i)
{
    if (i >= 0 && i <= static_cast<int>(Index_.size()))
    {
        IndexParticle_ = i;
    }
    else
    {
        std::cout << "No particle correspond to " << i << " index" << '\n';
    }
}

void Particle::SetAttribute(std::string const &name)
{
    if (FindParticle(name) == static_cast<int>(Index_.size()) + 1)
    {
        std::cout << "No particle corresponds to " << name << '\n';
    }
    else
    {
        IndexParticle_ = FindParticle(name);
    }
}

double Particle::GetEnergy() const
{
    double P2 = (Px_ * Px_) + (Py_ * Py_) + (Pz_ * Pz_);
    double m2 = GetMass() * GetMass();
    return std::sqrt(m2 + P2);
}

double Particle::InvMass(Particle &p) const
{
    double E2 = (GetEnergy() + p.GetEnergy()) * (GetEnergy() + p.GetEnergy());
    double Px = Px_ + p.GetPx();
    double Py = Py_ + p.GetPy();
    double Pz = Pz_ + p.GetPz();
    double P2 = Px * Px + Py * Py + Pz * Pz;
    return std::sqrt(E2 - P2);
}

void Particle::SetP(double Px, double Py, double Pz)
{
    Px_ = Px;
    Py_ = Py;
    Pz_ = Pz;
}

int Particle::Decay2Body(Particle &dau1, Particle &dau2) const
{
    if (GetMass() == 0.0)
    {
        std::cout << "Decayment cannot be performed if mass is zero" << '\n';
        return 1;
    }
    double massMot = GetMass();
    double massDau1 = dau1.GetMass();
    double massDau2 = dau2.GetMass();
    if (IndexParticle_ > -1)
    {
        float x1, x2, w, y1, y2;
        double invnum = 1. / RAND_MAX;
        do
        {
            x1 = 2.0 * rand() * invnum - 1.0;
            x2 = 2.0 * rand() * invnum - 1.0;
            w = x1 * x1 + x2 * x2;
        } while (w >= 1.0);
        w = std::sqrt((-2.0 * std::log(w)) / w);
        y1 = x1 * w;
        y2 = x2 * w;
        massMot += Index_[IndexParticle_]->GetWidth() * y1;
    }
    if (massMot < massDau1 + massDau2)
    {
        std::cout << "Decayment cannot be performed because mass is too low in this channel" << '\n';
        return 2;
    }
    double pout = std::sqrt((massMot * massMot - (massDau1 + massDau2) * (massDau1 + massDau2)) * (massMot * massMot - (massDau1 - massDau2) * (massDau1 - massDau2))) / massMot * 0.5;
    double norm = 2 * M_PI / RAND_MAX;
    double phi = rand() * norm;
    double theta = rand() * norm * 0.5 - M_PI / 2;
    dau1.SetP(pout * sin(theta) * cos(phi), pout * sin(theta) * sin(phi), pout * cos(theta));
    dau2.SetP(-pout * sin(theta) * cos(phi), -pout * sin(theta) * sin(phi), -pout * cos(theta));
    double energy = std::sqrt(Px_ * Px_ + Py_ * Py_ + Pz_ * Pz_ + massMot * massMot);
    double bx = Px_ / energy;
    double by = Py_ / energy;
    double bz = Pz_ / energy;
    dau1.Boost(bx, by, bz);
    dau2.Boost(bx, by, bz);
    return 0;
}

void Particle::PrintIndex()
{
    for (int i = 0; i != static_cast<int>(Index_.size()); i++)
    {
        std::cout << "Particle " << i << ':' << '\n';
        Index_[i]->Print();
    }
}

void Particle::PrintParticle()
{
    std::cout << "Particle name: " << Index_[IndexParticle_]->GetName() << '\n';
    std::cout << "Index position: " << IndexParticle_ << '\n';
    std::cout << "P: " << Px_ << "i " << Py_ << "j " << Pz_ << "k " << '\n';
}