// checks that the input string has 4 or 6 characters which are all digits

#include <string>
#include <cctype>
#include <algorithm>

bool validatePIN(std::string pin)
{
    if (pin.size() == 4 || pin.size() == 6)
    {
        return std::all_of(pin.begin(), pin.end(), [](char c) { return std::isdigit(c); });
    }
    else
    {
        return false;
    }
}