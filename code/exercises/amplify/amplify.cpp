// takes a number and returns a vector with the numbers from one to the one given 
// where all the multiples of 4 are multiplied by 10

#include <vector>
#include <numeric>

std::vector<int> amplify(int n)
{
    std::vector<int> result(n);
    std::iota(result.begin(), result.end(), 1);
    for (auto &v : result)
    {
        if (v % 4 == 0)
        {
            v *= 10;
        }
    }
    return result;
}