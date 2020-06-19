// returns a string where only the last 4 characters are visible

#include <string>

auto maskify(std::string str)
{
    if(str.size() <= 4)
    {
        return str;
    }
    else
    {
        for (auto it = str.begin(); it != str.end() - 4; ++it)
        {
            *it = '#';
        }
        return str;      
    }
}