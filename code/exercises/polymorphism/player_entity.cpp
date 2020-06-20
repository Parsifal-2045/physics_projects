#include <iostream>
#include <string>

class Entity
{
    public:
    virtual std::string GetName() {return "Entity"; }

};

class Player : public Entity
{
    private:
    std::string m_name;
    public:
    Player(const std::string &name) : m_name(name) {}
    std::string GetName() override {return m_name; }
};

void PrintName(Entity* entity)
{
    std::cout << entity->GetName() << '\n';
}

int main()
{
    Entity *e = new Entity();
    PrintName(e);
    Player *p = new Player("Luca");
    PrintName(p);
    Entity *entity = p;
    PrintName(entity);    
}