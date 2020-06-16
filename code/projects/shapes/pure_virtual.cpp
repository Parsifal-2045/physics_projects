#include <vector>
#include <memory>

class Point
{
    double x;
    double y;
};

struct Shape
{
    virtual ~Shape() = default;
    virtual Point where() const = 0;
};

struct Circle : Shape
{
    Point c;
    double r;
    Point where() const override
    {
        return c;
    }
};

struct Rectangle : Shape
{
    Point ul;
    Point lr;
    Point where() const override
    {
        return ul;
    }
};

Shape *create_shape(int i)
{
    if (i % 2)
    {
        return new Rectangle{};
    }
    else
    {
        return new Circle{};
    }
}

int main()
{
    std::vector<Shape *> shapes{create_shape(4), create_shape(3)};
    for (auto const &s : shapes)
    {
        s->where();
    }
    for (auto const &s : shapes)
    {
        delete s;
    }
}