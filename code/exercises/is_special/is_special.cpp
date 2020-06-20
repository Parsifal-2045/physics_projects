// returns true if all of the elements in even positions are even and all those in odd positions are odd

#include <vector>

bool isSpecialArray(std::vector<int> arr)
{
    int i = 0;
    bool even;
    bool odd;
    for (auto it = arr.begin(); it != arr.end(); it++)
    {
        if (*it % 2 == 0)
        {
            even = i % 2 == 0;
            ++i;
            if (even == false)
            {
                return false;
            }
        }
        if (*it % 2 == 1)
        {
            odd = i % 2 == 1;
            ++i;
            if (odd == false)
            {
                return false;
            }
        }
    }
    return true;
}

// simpler version

bool isSpecialArray(std::vector<int> arr)
{
    for (int i = 0; i < arr.size(); i++)
    {
        if (i % 2 != arr[i] % 2)
        {
            return false;
        }
    }
    return true;
}