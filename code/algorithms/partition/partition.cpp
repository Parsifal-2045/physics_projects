#include <fmt/core.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

struct Particle {
    float mass;
    std::string name;
    std::string type;
};

std::ostream& operator<<(std::ostream& os, Particle const& p) {
    os << p.name;
    return os;
}

void printParticles(std::vector<Particle> const& v, const bool verbose) {
    std::cout << "[ ";
    for (auto p : v) {
        std::cout << '(';
        std::cout << p;
        if (verbose) {
            std::cout << ", " << p.type << ", " << p.mass;
        }
        std::cout << "), ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    auto particles = std::vector<Particle>{
        Particle{938.f, "proton", "fermion"}, Particle{0.f, "photon", "boson"},
        Particle{939.f, "neutron", "fermion"},
        Particle{0.5f, "electron", "fermion"},
        Particle{125000.f, "higgs", "boson"}
        /* many other particles */
    };

    // 1) Find lightest particle
    auto lessMass = [](Particle const& p1, Particle const& p2) {
        return p1.mass < p2.mass;
    };
    auto lightestParticle = std::ranges::min_element(particles, lessMass);
    std::cout << "The lightest particle is: " << lightestParticle->name
              << ", a " << lightestParticle->type << " with mass "
              << lightestParticle->mass << std::endl;

    // 2) Put all bosons to the beginning of the vector
    bool verbose = false;
    std::cout << "Unsorted vector of particles: " << std::endl;
    printParticles(particles, verbose);

    std::ranges::partition(particles,
                           [](Particle const& p) { return p.type == "boson"; });

    std::cout << "Sorted vector of particles: " << std::endl;
    printParticles(particles, verbose);
}
