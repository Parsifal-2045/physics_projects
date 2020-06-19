//returns a string where letters are in alphabetical order

#include <string>
#include <algorithm>

std::string alphabetSoup(std::string str) 
{
    std::sort(str.begin(), str.end(), [] (char a, char b) {return static_cast<int>(a) < static_cast<int>(b);});
    return str;   	
}