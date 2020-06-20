// returns true if the value (val) is present in each of the vectors at least once

#include <vector>

bool isOmnipresent(std::vector<std::vector<int>> arr, int val)
{
    int count = 0;
    for (auto &vector : arr)
    {
        for (auto v : vector)
        {
            if (v == val)
            {
                ++count;
                break;
            }
        }
    }
    return count == arr.size();
}