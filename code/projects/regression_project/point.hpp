#ifndef POINT_HPP
#define POINT_HPP

struct Point
{
    double x;
    double y;
};

inline bool operator==(Point const &lhs, Point const &rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

inline bool operator!=(Point const &lhs, Point const &rhs)
{
    return !(lhs == rhs);
}

inline Point operator+(Point const &lhs, Point const &rhs)
{
    return Point{lhs.x + rhs.x, lhs.y + rhs.y};
}

#endif