//returns a vector of strings with only words made of 4 or more letters

#include <vector>

auto isFourLetters(std::vector<std::string> &v)
{
    std::vector<std::string> result;
    for (auto string : v)
    {
        if (std::distance(string.begin(), string.end()) == 4)
        {
            result.push_back(string);
        }       
    }
    return result;
}