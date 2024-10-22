#ifndef NEW_CACHING_ALLOCATOR_H
#define NEW_CACHING_ALLOCATOR_H

#include <cassert>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <optional>

// constants
#define MAX_BINS 7
#define MIN_BINS 3
#define GROWTH 8
#define CACHING true
#define DEBUG false

namespace detail
{

    inline constexpr unsigned int power(unsigned int base, unsigned int exponent)
    {
        unsigned int power = 1;
        while (exponent > 0)
        {
            if (exponent & 1)
            {
                power = power * base;
            }
            base = base * base;
            exponent = exponent >> 1;
        }
        return power;
    }

    // format a memory size in B/kB/MB/GB
    inline std::string as_bytes(size_t value)
    {
        if (value == std::numeric_limits<size_t>::max())
        {
            return "unlimited";
        }
        else if (value >= (1 << 30) and value % (1 << 30) == 0)
        {
            return std::to_string(value >> 30) + " GB";
        }
        else if (value >= (1 << 20) and value % (1 << 20) == 0)
        {
            return std::to_string(value >> 20) + " MB";
        }
        else if (value >= (1 << 10) and value % (1 << 10) == 0)
        {
            return std::to_string(value >> 10) + " kB";
        }
        else
        {
            return std::to_string(value) + "  B";
        }
    }

} // namespace detail

class CachingAllocator
{
public:
    // constructor
    CachingAllocator(unsigned int binGrowth,
                     unsigned int minBin,
                     unsigned int maxBin,
                     const bool cacheAllocations,
                     const bool debug)
        : binGrowth_(binGrowth),
          minBin_(minBin),
          maxBin_(maxBin),
          minBinBytes_(detail::power(binGrowth, minBin)),
          maxBinBytes_(detail::power(binGrowth, maxBin)),
          cacheAllocations_(cacheAllocations),
          debug_(debug)
    {
        if (debug_)
        {
            std::ostringstream out;
            out << "Caching Allocator settings\n"
                << "  bin growth " << binGrowth_ << "\n"
                << "  min bin    " << minBin_ << "\n"
                << "  max bin    " << maxBin_ << "\n"
                << "  resulting bins:\n";
            for (auto bin = minBin_; bin <= maxBin_; ++bin)
            {
                auto binSize = detail::power(binGrowth_, bin);
                out << "    " << std::right << std::setw(12) << detail::as_bytes(binSize) << '\n';
            }
            std::cout << out.str() << std::endl;
        }
    }

    // destructor
    ~CachingAllocator()
    {
        {
            // the destructor should never be called when some blocks are still live
            std::scoped_lock lock(mutex_);
            assert(liveBlocks_.empty());
            assert(cachedBytes_.live == 0);
        }
        // release all the cached blocks
        free();
    }

    // allocate given number of bytes
    void *allocate(size_t bytes)
    {
        // create a block for the requested allocation
        MemoryBlock block;
        block.bytes_requested = bytes;
        std::tie(block.bin, block.bytes) = findBin(bytes);

        // try to re-use a cached block, or allocate a new buffer
        if (not tryReuseCachedBlock(block))
        {
            allocateNewBlock(block);
        }

        return block.ptr;
    }

    // deallocate memory block and cache it for later
    void deallocate(void *ptr)
    {
        std::scoped_lock lock(mutex_);

        auto blockIterator = liveBlocks_.find(ptr);
        if (blockIterator == liveBlocks_.end())
        {
            std::stringstream ss;
            ss << "Trying to free a non-live block at " << ptr;
            throw std::runtime_error(ss.str());
        }

        // remove the block from the list of live blocks
        MemoryBlock block = std::move(blockIterator->second);
        liveBlocks_.erase(blockIterator);
        cachedBytes_.live -= block.bytes;
        cachedBytes_.requested -= block.bytes_requested;

        bool recache = cacheAllocations_;
        if (recache)
        {
            cachedBytes_.free += block.bytes;
            block.readyToReuse_ = true;
            cachedBlocks_.insert(std::make_pair(block.bin, block));

            if (debug_)
            {
                std::ostringstream out;
                out << " Returned " << block.bytes << " bytes at " << ptr << " .\n\t\t " << cachedBlocks_.size()
                    << " available blocks cached (" << cachedBytes_.free << " bytes), " << liveBlocks_.size()
                    << " live blocks (" << cachedBytes_.live << " bytes) outstanding." << std::endl;
                std::cout << out.str() << std::endl;
            }
        }
        else
        {
            std::free(block.ptr);

            if (debug_)
            {
                std::ostringstream out;
                out << " Freed " << block.bytes << " bytes at " << ptr << " .\n\t\t " << cachedBlocks_.size()
                    << " available blocks cached (" << cachedBytes_.free << " bytes), " << liveBlocks_.size()
                    << " live blocks (" << cachedBytes_.live << " bytes) outstanding." << std::endl;
                std::cout << out.str() << std::endl;
            }
        }
    }

    void free()
    {
        std::scoped_lock lock(mutex_);

        while (not cachedBlocks_.empty())
        {
            auto blockIterator = cachedBlocks_.begin();
            MemoryBlock block = std::move(blockIterator->second);
            cachedBlocks_.erase(blockIterator);
            cachedBytes_.free -= block.bytes;
            std::free(block.ptr);

            if (debug_)
            {
                std::ostringstream out;
                out << "Freed " << block.bytes << " bytes.\n\t\t  " << cachedBlocks_.size() << " available blocks cached ("
                    << cachedBytes_.free << " bytes), " << liveBlocks_.size() << " live blocks (" << cachedBytes_.live
                    << " bytes) outstanding." << std::endl;
                std::cout << out.str() << std::endl;
            }
        }
    }

private:
    // counter for total free, live and requested bytes
    struct CachedBytes
    {
        size_t free = 0;      // total bytes freed and cached
        size_t live = 0;      // total bytes currently in use
        size_t requested = 0; // total bytes requested and currently in use
    };

    // reusable memory buffers
    struct MemoryBlock
    {
        std::byte *ptr;             // pointer to data
        size_t bytes = 0;           // bytes allocated
        size_t bytes_requested = 0; // bytes requested
        unsigned int bin = 0;       // bin class id, binGrowth^bin is the block size
        bool readyToReuse_ = false; // caching status
    };

    std::mutex mutex_;

    const unsigned int binGrowth_; // bin growth factor;
    const unsigned int minBin_;    // the smallest bin is set to binGrowth^minBin bytes;
    const unsigned int maxBin_;    // the largest bin is set to binGrowth^maxBin bytes;
                                   // larger allocations will fail;
    const size_t minBinBytes_;     // bytes of the smallest bin
    const size_t maxBinBytes_;     // bytes of the biggest bin
    const bool cacheAllocations_;  // caching policy
    const bool debug_;             // prints debug information

    using CachedBlocks = std::multimap<unsigned int, MemoryBlock>;
    using BusyBlocks = std::map<void *, MemoryBlock>;

    CachedBytes cachedBytes_;
    CachedBlocks cachedBlocks_; // set of cached allocations available for reuse
    BusyBlocks liveBlocks_;     // map of pointers to the live allocations currently in use

    // return (bin, bin size)
    std::pair<unsigned int, size_t> findBin(size_t bytes) const
    {
        if (bytes < minBinBytes_)
        {
            return std::make_pair(minBin_, minBinBytes_);
        }
        if (bytes > maxBinBytes_)
        {
            throw std::runtime_error("Requested allocation size " + std::to_string(bytes) +
                                     " bytes is too large for the caching detail with maximum bin " +
                                     std::to_string(maxBinBytes_) +
                                     " bytes. You might want to increase the maximum bin size");
        }

        unsigned int bin = minBin_;
        size_t binBytes = minBinBytes_;
        while (binBytes < bytes)
        {
            ++bin;
            binBytes *= binGrowth_;
        }

        return std::make_pair(bin, binBytes);
    }

    bool tryReuseCachedBlock(MemoryBlock &block)
    {
        std::scoped_lock lock(mutex_);

        // iterate through the range of cached blocks in the same bin
        const auto [begin, end] = cachedBlocks_.equal_range(block.bin);
        for (auto blockIterator = begin; blockIterator != end; ++blockIterator)
        {
            if (cacheAllocations_ or
                blockIterator->second.readyToReuse_)
            {
                block = blockIterator->second;

                block.readyToReuse_ = false;

                // insert the cached block into the live blocks
                liveBlocks_[block.ptr] = block;

                // update total sizes information
                cachedBytes_.free -= block.bytes;
                cachedBytes_.live += block.bytes;
                cachedBytes_.requested += block.bytes_requested;

                if (debug_)
                {
                    std::ostringstream out;
                    out << "Reused cached block at " << block.ptr << " (" << block.bytes << " bytes)" << std::endl;
                    std::cout << out.str();
                }

                // remove the reused block from the list of cached blocks
                cachedBlocks_.erase(blockIterator);
                return true;
            }
        }

        return false;
    }

    std::byte *allocateBuffer(size_t bytes)
    {
        if (debug_)
        {
            std::ostringstream out;
            out << "\tNew allocation of " << bytes << " bytes." << std::endl;
            std::cout << out.str() << std::endl;
        }
        return static_cast<std::byte *>(std::aligned_alloc(std::hardware_destructive_interference_size, bytes));
    }

    void allocateNewBlock(MemoryBlock &block)
    {
        block.ptr = allocateBuffer(block.bytes);
        if (block.ptr == nullptr)
        {
            // the allocation attempt failed: free all cached blocks and retry
            if (debug_)
            {
                std::ostringstream out;
                out << "Failed to allocate " << block.bytes << " bytes,"
                    << " retrying after freeing cached allocations" << std::endl;
                std::cout << out.str() << std::endl;
            }
            free();

            block.ptr = allocateBuffer(block.bytes);
        }

        // set newly allocated block to not be ready to be reused
        block.readyToReuse_ = false;

        {
            std::scoped_lock lock(mutex_);
            cachedBytes_.live += block.bytes;
            cachedBytes_.requested += block.bytes_requested;
            liveBlocks_[block.ptr] = block;
        }

        if (debug_)
        {
            std::ostringstream out;
            out << "Allocated new block at " << block.ptr << " of " << block.bytes << " bytes" << std::endl;
            std::cout << out.str() << std::endl;
        }
    }
};

#endif