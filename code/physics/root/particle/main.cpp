#include <iostream>
#include <string>

class ParticleType
{
private:
    std::string const name_;
    double const mass_;
    int const charge_;

public:
    ParticleType(std::string const &name = {}, double const &mass = {}, int const &charge = {})
        : name_{name}, mass_{mass}, charge_{charge} {}
    std::string const GetName() { return name_; };
    double const GetMass() { return mass_; };
    int const GetCharge() { return charge_; };
    virtual void print() const
    {
        std::cout << "Particle name: " << name_ << '\n';
        std::cout << "Particle mass: " << mass_ << '\n';
        std::cout << "Particle charge: " << charge_ << '\n';
    }
};

class ResonanceType : public ParticleType
{
private:
    std::string const name_;
    double const mass_;
    int const charge_;
    double const width_;

public:
    ResonanceType(std::string const &name = {}, double const &mass = {}, int const &charge = {},
                  double const &width = {}) : name_{name}, mass_{mass}, charge_{charge}, width_{width} {}
    double const GetWidth() { return width_; };
    void print() const override
    {
        ParticleType::print();
        std::cout << "Resonance width: " << width_ << '\n';
    }
};

int main()
{
    ParticleType a("Proton", 12.6, 1);
    a.print();
    ResonanceType b;
    b.print();
}