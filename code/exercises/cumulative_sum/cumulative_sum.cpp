// returns a vector where each element is replaced with the sum of the elements until itself

#include <vector>

std::vector<int> cumulativeSum(std::vector<int> array)
{
    if(array == std::vector<int>{})
    {
        return {};
    }
    else
    {
        std::vector<int> result;
        for (auto it = array.begin() + 1; it <= array.end(); ++it)
        {
        result.push_back(std::accumulate(array.begin(), it, 0));
        }
        return result;
    }
}