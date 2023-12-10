/*

In the code below, replace inheritance with the use of a std::variant.
Two solutions are provided :
1. with `std::get_if`,
2. with `std::visit`.

*/

#include <iostream>
#include <memory>
#include <vector>
#include <variant>

struct Electron
{
  static void print() { std::cout << "E" << std::endl; }
};

struct Proton
{
  static void print() { std::cout << "P" << std::endl; }
};

struct Neutron
{
  static void print() { std::cout << "N" << std::endl; }
};

void print_if(std::variant<Electron, Proton, Neutron> const &p)
{
  if (std::get_if<Electron>(&p))
  {
    Electron::print();
  }
  else if (std::get_if<Proton>(&p))
  {
    Proton::print();
  }
  else if (std::get_if<Neutron>(&p))
  {
    Neutron::print();
  }
  else
  {
    throw std::runtime_error("No particle of this type exists");
  }
}

int main()
{
  std::vector<std::variant<Electron, Proton, Neutron>> ps{Electron{}, Proton{}, Neutron{}};

  for (auto const &p : ps)
  {
    print_if(p);
  }
}
