#include <iostream>
#include <string>

class Printable
{
    public:
    virtual std::string GetClassName() = 0;
};

class Entity : public Printable
{
    public:
    virtual std::string GetName() {return "Entity"; }
    std::string GetClassName() override {return "Entity"; }

};

class Player : public Entity
{
    private:
    std::string m_name;
    public:
    Player(const std::string &name) : m_name(name) {}
    std::string GetName() override {return m_name; }
    std::string GetClassName() override {return "Player"; }
};

void PrintName(Entity* entity)
{
    std::cout << entity->GetName() << '\n';
}

void Print(Printable *obj)
{
    std::cout << obj->GetClassName() << '\n';
}

int main()
{
    Entity *e = new Entity();
    Print(e);
    Player *p = new Player("Luca");
    Print(p);
}