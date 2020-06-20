// returns a string with only the letters from the input string

#include <string>
#include <cctype>
#include <algorithm>

std::string lettersOnly(std::string str)
{
    str.erase(std::remove_if(str.begin(), str.end(), [] (char c) {return !(isalpha(c));}), str.end());
    return str;
}