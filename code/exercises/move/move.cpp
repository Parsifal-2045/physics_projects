//returns a string where each letter is exchanged with the next one

#include <string>
#include <algorithm>

auto move(std::string word)
{
    std::transform(word.begin(), word.end(), word.begin(), [] (char c) {return c + 1;});
    return word;
}