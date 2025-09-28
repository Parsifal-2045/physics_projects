# Compiler settings
CXX      := g++

CXXFLAGS := -Wall -Wextra -O2


# Default rule
all: matrixMultiplication matrixMultiplicationTranspose

matrixMultiplication: matrixMultiplication.cc 
	$(CXX) $(CXXFLAGS) $^ -o $@.out

matrixMultiplicationTranspose: matrixMultiplication.cc 
	$(CXX) $(CXXFLAGS) -DTRANSPOSE $^ -o $@.out

clean:
	rm -f *.out
