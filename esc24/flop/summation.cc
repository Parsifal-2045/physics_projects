/*

NAME:
   summation (the C++ version)

Purpose:
   Create an array of numbers and explore floating point
   arithmetic issues as you sum the array

Usage:
   This program depends on a set of functions from the file
   UtilityFunctions.c.   For reasons of pedagogy, do not look
   at the functions in UtilityFunctions.c. Treat it as a block box.

        g++ summation.cc UtilityFunctions.c

History:
   Written by Tim Mattson, 9/2023.

*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cfenv>
#include "UtilityFunctions.h" // where FillSequence comes from

constexpr int N = 100000; // length of sequence of numbers to work with

//============================================================================

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
   os << "[ ";
   for (auto &value : v)
   {
      os << value << ' ';
   }
   os << ']' << '\n';
   return os;
}

// accumulate with Kahan algorithm
float KahanAccumulate(const std::vector<float> &vector)
{
   if (vector.size() < 1)
   {
      return -1;
   }

   float correction = 0.0f;
   float ksum = 0.0f;
   for (auto &v : vector)
   {
      float xcorrected = v - correction;
      float tmpSum = ksum + xcorrected;
      correction = (tmpSum - ksum) - xcorrected;
      ksum = tmpSum;
   }
   return ksum;
}

int main()
{
   // change rounding mode.  Options are:
   //         FE_DOWNWARD,   FE_TOWARDZERO,
   //         FE_UPWARD,          FE_TONEAREST
   std::fesetround(FE_TONEAREST); // default

   std::vector<float> seq(N); // Sequence to sum
   float True_sum;            // An estimate of the actual sum

   FillSequence(N, seq.data(), &True_sum); // populate seq with N random values > 0

   float sum = std::accumulate(seq.begin(), seq.end(), 0.0f);

   float kahan_sum = KahanAccumulate(seq); // no sorting required

   // sort might help the regular sum, but sorting is expensive
   std::sort(seq.begin(), seq.end());
   float sorted_sum = std::accumulate(seq.begin(), seq.end(), 0.0f);

   std::sort(seq.begin(), seq.end(), [](float i, float j)
             { return j < i; });
   float reverse_sort_sum = std::accumulate(seq.begin(), seq.end(), 0.0f);

   printf("True sum: %f\n", True_sum);
   printf("Sum: %f\n", sum);
   printf("Sorted sum: %f\n", sorted_sum);
   printf("Reverse-sorted sum: %f\n", reverse_sort_sum);
   printf("Kahan sum: %f\n", kahan_sum);
   std::cout << "Naive sum error: " << True_sum - sum << '\n'
             << "Sorted sum error: " << True_sum - sorted_sum << '\n'
             << "Reverse sorted sum error: " << True_sum - reverse_sort_sum << '\n'
             << "Kahan sum error: " << True_sum - kahan_sum << '\n';
}
