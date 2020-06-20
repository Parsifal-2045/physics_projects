// returns a vector whose first element is the count of positives in the input
// and the second is the sum of the negatives

#include <vector>

std::vector<int> countPosSumNeg(std::vector<int> arr)
{
    if (arr == std::vector<int>{})
    {
        return arr;
    }
    else
    {
        int negative = 0;
        int positive = 0;
        for (auto value : arr)
        {
            if (value <= 0)
            {
                negative += value;
            }
            if (value > 0)
            {
                positive++;
            }
        }
        std::vector<int> result{positive, negative};
        return result;
    }
}