// takes a vector with nine numbers between 1 and 10 and returns the missing one

#include <vector>
#include <algorithm>

int missingNum(std::vector<int> arr)
{
    std::sort(arr.begin(), arr.end());
    if (arr[0] == 2)
    {
        return 1;
    }
    else
    {
        for (auto it = arr.begin(); it != arr.end(); ++it)
        {
            if (*it + 1 != *(it + 1))
            {
                return *it + 1;
            }
        }
    }
    return 0;
}