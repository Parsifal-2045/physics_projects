#include "particle_type.hpp"
#include "resonance_type.hpp"

int main()
{
    ParticleType a{"Proton", 12.6, 1};
    a.print();
    ResonanceType b{"test 1", 25, 4, 1};
    b.print();
}