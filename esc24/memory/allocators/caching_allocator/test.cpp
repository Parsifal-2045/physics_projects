#include "ca.h"

int main()
{
    CachingAllocator allocator{GROWTH, MIN_BINS, MAX_BINS, CACHING, DEBUG};
}