#include <vector>
#include <memory>

class Point
{
    double x;
    double y;
};

struct Shape
{
    Point p;
    Shape(Point p) : p{p} {}
    virtual ~Shape() = default;
    virtual Point where() const
    {
        return p;
    }
};

struct Circle : Shape
{
    double r;
    Circle(Point p, double d) : Shape{p}, r{d} {}
    //where is now inherited from Shape
};

struct Rectangle : Shape
{
    Point lr;
    Rectangle(Point p1, Point p2) : Shape{p1}, lr{p2} {}
    Point where() const override
    {
        return (Shape::where() + lr) / 2;
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