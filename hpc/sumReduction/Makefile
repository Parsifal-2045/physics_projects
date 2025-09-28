# Compiler settings
CXX      := g++

CXXFLAGS := -fopenmp -Wall -Wextra -O2


# Default rule
all: sum

sum: sum.cc 
	$(CXX) $(CXXFLAGS) $^ -o $@.out

clean:
	rm -f sum.out
