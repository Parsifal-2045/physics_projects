#include "Particle.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <cassert>

// utility class for timing
class Timer
{
public:
  Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}

  void reset() { start_time_ = std::chrono::high_resolution_clock::now(); }

  double elapsed() const
  {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time =
        end_time - start_time_;
    return elapsed_time.count();
  }

private:
  std::chrono::high_resolution_clock::time_point start_time_;
};

// function to initialize AoS
void initializeAoS(
    std::vector<GoodParticle> &particles,
    const int nParticles,
    const std::vector<double> &pxDist,
    const std::vector<double> &pyDist,
    const std::vector<double> &pzDist,
    const std::vector<double> &xDist,
    const std::vector<double> &yDist,
    const std::vector<double> &zDist,
    const std::vector<double> &massDist,
    const std::vector<double> &energyDist)
{
  for (int i = 0; i != nParticles; ++i)
  {
    particles.emplace_back(massDist[i],
                           energyDist[i],
                           pxDist[i],
                           pyDist[i],
                           pzDist[i],
                           xDist[i],
                           yDist[i],
                           zDist[i],
                           0,
                           false,
                           false,
                           false,
                           "test_particle");
  }
}

// function for computation on AoS
void calculateAoS(std::vector<GoodParticle> &particles,
                  double x_max)
{
  std::mt19937 rng(2);
  std::uniform_real_distribution<double> dist(0., 50.);

  for (auto &particle : particles)
  {
    double timeInterval = dist(rng);
    double newX = particle.getX() + particle.getPx() / particle.getMass() * timeInterval;
    particle.setX(newX);
    bool hitX = newX < 0 or newX > x_max;
    particle.setCollisionX(hitX);
    if (!hitX)
    {
      particle.setPx(-particle.getPx());
    }
  }
}

// function to initialize SoA
void initializeSoA(ParticleSoA &particles,
                   const int nParticles,
                   const std::vector<double> &pxDist,
                   const std::vector<double> &pyDist,
                   const std::vector<double> &pzDist,
                   const std::vector<double> &xDist,
                   const std::vector<double> &yDist,
                   const std::vector<double> &zDist,
                   const std::vector<double> &massDist,
                   const std::vector<double> &energyDist)
{
  particles.setMass(massDist);
  particles.setEnergy(energyDist);
  particles.setPx(pxDist);
  particles.setPy(pyDist);
  particles.setPz(pzDist);
  particles.setX(xDist);
  particles.setY(yDist);
  particles.setZ(zDist);
  particles.setId(0);
  particles.setCollisions(false);
  particles.setName("test_particle");
}

// function for computation on SoA
void calculateSoA(std::vector<double> &x, std::vector<double> &px, const std::vector<double> &mass, std::vector<bool> hitX, const double x_max)
{
  std::mt19937 rng(2);
  std::uniform_real_distribution<double> dist(0., 50.);

  for (size_t i = 0; i != x.size(); ++i)
  {
    double timeInterval = dist(rng);
    x[i] = x[i] + px[i] / mass[i] * timeInterval;
    hitX.emplace_back(x[i] < 0 or x[i] > x_max);
    if (!hitX[i])
    {
      px[i] = -px[i];
    }
  }
}

int main()
{

  std::cout << "sizeof(GoodParticle) " << sizeof(GoodParticle) << std::endl;

  std::random_device rd;
  std::mt19937 gen(2);

  std::vector<double> pxDist;
  std::vector<double> pyDist;
  std::vector<double> pzDist;
  std::vector<double> xDist;
  std::vector<double> yDist;
  std::vector<double> zDist;
  std::vector<double> massDist;
  std::vector<double> energyDist;

  constexpr int NIter = 1000000;
  constexpr double x_max = 200;
  constexpr int Npart = 1 << 10;

  auto fillVec = [&](std::vector<double> &vec, double min, double max)
  {
    vec.reserve(Npart);
    std::uniform_real_distribution<double> dist(min, max);
    std::generate_n(std::back_inserter(vec), Npart, [&]
                    { return dist(gen); });
  };

  fillVec(pxDist, 10., 100.);
  fillVec(pyDist, 10., 100.);
  fillVec(pzDist, 10., 100.);
  fillVec(xDist, -100., 100.);
  fillVec(yDist, -100., 100.);
  fillVec(zDist, -300., 300.);
  fillVec(massDist, 10., 100.);
  energyDist.reserve(Npart);

  for (int i = 0; i != Npart; ++i)
  {
    energyDist.emplace_back(std::sqrt(pxDist[i] * pxDist[i] + pyDist[i] * pyDist[i] + pzDist[i] * pzDist[i] + massDist[i] * massDist[i]));
  }

  auto const t = std::uniform_real_distribution<double>(-2., 2.)(gen);

  Timer timer; // starting the clock
  std::vector<GoodParticle> good_particles;
  good_particles.reserve(Npart);

  initializeAoS(good_particles, Npart, pxDist, pyDist, pzDist, xDist, yDist, zDist, massDist, energyDist);

  calculateAoS(good_particles, x_max);

  std::cout << "AoS Elapsed time: " << timer.elapsed() << " ms " << "\n";

  timer.reset();
  ParticleSoA soa_particles(Npart);
  initializeSoA(soa_particles, Npart, pxDist, pyDist, pzDist, xDist, yDist, zDist, massDist, energyDist);

  calculateSoA(soa_particles.getX(), soa_particles.getPx(), soa_particles.getMass(), soa_particles.getCollisionX(), x_max);
  std::cout << "SoA Elapsed time: " << timer.elapsed() << " ms " << "\n";
}
