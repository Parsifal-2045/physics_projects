#include <iostream>

struct vector {
    double x;
    double y;
    double z;
};

bool operator==(vector const& lhs, vector const& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

vector operator+ (vector const& lhs, vector const& rhs) 
{
    return vector{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

vector operator- (vector const& lhs, vector const&rhs)
{
    return vector{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

int main()
{
    vector a = {1, 1, 1};
    vector b = {1, 2, 3};
    a + b;
}

/* vector sum(vector v1, vector v2)
{
    vector w = v1 + v2;
    return w;
}

vector diff(vector v1, vector v2)
{
    vector w = v1 - v2;
    return w;
} */
