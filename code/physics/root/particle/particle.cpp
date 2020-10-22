#include <iostream>
#include <vector>
#include <cmath>
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

Particle::Particle(std::string const &name, double Px = 0, double Py = 0, double Pz = 0)
{
    Px_ = Px;
    Py_ = Py;
    Pz_ = Pz;
    IndexParticle_ = FindParticle(name);
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
            std::cout << name << " already added" << '\n';
        }
    }
    else
    {
        std::cout << "Maximum number of Particle Types reached" << '\n';
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

double Particle::TotalEnergy() const
{
    double P2 = (Px_ * Px_) + (Py_ * Py_) + (Pz_ * Pz_);
    double m2 = Index_[IndexParticle_]->GetMass() * Index_[IndexParticle_]->GetMass();
    return std::sqrt(m2 + P2);
}

double Particle::InvMass(Particle &p) const
{
    double P2 = (Px_ * Px_) + (Py_ * Py_) + (Pz_ * Pz_);
    double m2 = Index_[IndexParticle_]->GetMass() * Index_[IndexParticle_]->GetMass();
    double E1 = std::sqrt(m2 + P2);
    double E2 = p.TotalEnergy();
    double Px2 = (Px_ + p.GetPx()) * (Px_ + p.GetPx());
    double Py2 = (Py_ + p.GetPy()) * (Py_ + p.GetPy());
    double Pz2 = (Pz_ + p.GetPz()) * (Pz_ + p.GetPz());
    return std::sqrt(((E1 + E2) * (E1 + E2)) - (Px2 + Py2 + Pz2));
}

void Particle::SetP(double Px, double Py, double Pz)
{
    Px_ = Px;
    Py_ = Py;
    Pz_ = Pz;
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
    std::cout << "Index: " << IndexParticle_ << '\n';
    std::cout << "Name: " << Index_[IndexParticle_]->GetName() << '\n';
    std::cout << "P: " << Px_ << "i " << Py_ << "j " << Pz_ << "k " << '\n';
}