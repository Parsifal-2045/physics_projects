#ifndef GOOD_PARTICLE
#define GOOD_PARTICLE
#include <string>
#include <vector>
#include <iostream>


class GoodParticle
{
public:
  GoodParticle(float mass,
               float energy,
               double px,
               double py,
               double pz,
               double x,
               double y,
               double z,
               int id,
               bool collisionX,
               bool collisionY,
               bool collisionZ,
               std::string const &name) : mass_(mass),
                                          energy_(energy),
                                          px_(px_),
                                          py_(py),
                                          pz_(pz),
                                          x_(x),
                                          y_(y),
                                          z_(z),
                                          id_(id),
                                          collisionX_(collisionX),
                                          collisionY_(collisionY),
                                          collisionZ_(collisionZ),
                                          name_(name) {}

  const double getX() const
  {
    return x_;
  }

  const double getPx() const
  {
    return px_;
  }
  const double getMass() const
  {
    return mass_;
  }

  void setX(const double x)
  {
    x_ = x;
  }

  void setPx(const double px)
  {
    px_ = px;
  }

  void setCollisionX(bool hit)
  {
    collisionX_ = hit;
  }

private:
  const float mass_, energy_;                 // 8
  double px_, py_, pz_;                       // 32
  double x_, y_, z_;                          // 56
  int id_;                                    // 60
  bool collisionX_, collisionY_, collisionZ_; // 63 + 1 padding
  const std::string name_;                    // 32
};

class ParticleSoA
{
public:
  ParticleSoA(int N)
  {
    mass_.reserve(N);
    energy_.reserve(N);
    px_.reserve(N);
    py_.reserve(N);
    pz_.reserve(N);
    x_.reserve(N);
    y_.reserve(N);
    z_.reserve(N);
    id_.reserve(N);
    collisionX_.reserve(N);
    collisionY_.reserve(N);
    collisionZ_.reserve(N);
    name_.reserve(N);
  };

  std::vector<double> &getX()
  {
    return x_;
  }

  std::vector<double> &getPx()
  {
    return px_;
  }

  std::vector<double> &getMass()
  {
    return mass_;
  }
  std::vector<bool> &getCollisionX()
  {
    return collisionX_;
  }

  void setMass(const std::vector<double> &mass)
  {
    mass_ = mass;
  }

  void setEnergy(const std::vector<double> &energy)
  {
    energy_ = energy;
  }

  void setPx(const std::vector<double> &px)
  {
    px_ = px;
  }

  void setPy(const std::vector<double> &py)
  {
    py_ = py;
  }

  void setPz(const std::vector<double> &pz)
  {
    pz_ = pz;
  }

  void setX(const std::vector<double> &x)
  {
    x_ = x;
  }

  void setY(const std::vector<double> &y)
  {
    y_ = y;
  }

  void setZ(const std::vector<double> &z)
  {
    z_ = z;
  }

  void setId(const int id)
  {
    std::fill(id_.begin(), id_.end(), id);
  }

  void setCollisions(const bool hit)
  {
    std::fill(collisionX_.begin(), collisionX_.end(), hit);
    std::fill(collisionY_.begin(), collisionY_.end(), hit);
    std::fill(collisionZ_.begin(), collisionZ_.end(), hit);
  }

  void setName(const std::string &name)
  {
    std::fill(name_.begin(), name_.end(), name);
  }

private:
  std::vector<double> mass_, energy_;
  std::vector<double> px_, py_, pz_;
  std::vector<double> x_, y_, z_;
  std::vector<int> id_;
  std::vector<bool> collisionX_, collisionY_, collisionZ_;
  std::vector<std::string> name_;
};
#endif
