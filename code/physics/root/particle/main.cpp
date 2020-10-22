#include <iostream>
#include <vector>
#include "particle_type.hpp"
#include "resonance_type.hpp"
#include "particle.hpp"

int main()
{
    Particle::AddParticleType("Test 0", 0, 0, 0);
    Particle::AddParticleType("Test 1", 1, 1, 1);
    Particle::AddParticleType("Test 2", 2, 2, 2);
    Particle::AddParticleType("Test 3", 3, 3, 3);
    Particle::AddParticleType("Test 4", 4, 4, 4);
    Particle::AddParticleType("Test 5", 5, 5, 5);
    Particle::AddParticleType("Test 6", 6, 6, 6);
    Particle::AddParticleType("Test 7", 7, 7, 7);
    Particle::AddParticleType("Test 8", 8, 8, 8);
    Particle::AddParticleType("Test 9", 9, 9, 9);
    Particle p{"Test 0", 0, 0, 0};
    {
        Particle p2{"Test 2", 2, 2, 2};
    }
    Particle::PrintIndex();
    p.PrintParticle();
}

/*  Set of 10 Particles for testing

    Particle::AddParticleType("Test 0", 0, 0, 0);
    Particle::AddParticleType("Test 1", 1, 1, 1);
    Particle::AddParticleType("Test 2", 2, 2, 2);
    Particle::AddParticleType("Test 3", 3, 3, 3);
    Particle::AddParticleType("Test 4", 4, 4, 4);
    Particle::AddParticleType("Test 5", 5, 5, 5);
    Particle::AddParticleType("Test 6", 6, 6, 6);
    Particle::AddParticleType("Test 7", 7, 7, 7);
    Particle::AddParticleType("Test 8", 8, 8, 8);
    Particle::AddParticleType("Test 9", 9, 9, 9);
*/