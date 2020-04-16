#ifndef OPTIONAL_EXTRACT_HPP
#define OPTIONAL_EXTRACT_HPP
#include <iostream>

struct optional_extract
{
    char c;
    optional_extract(char c) : c{c} {}
};

std::istream &operator>>(std::istream &ins, optional_extract e)
{
    // Skip leading whitespace IFF user is not asking to extract a whitespace character
    if (!std::isspace(e.c))
        ins >> std::ws;

    // Attempt to get the specific character
    if (ins.peek() == e.c)
        ins.get();

    // There is no failure!
    return ins;
}

#endif