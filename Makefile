CXX_FLAGS=-std=c++20 -O2 -Wall -Wextra -Wpedantic -fopenmp

all: simulation_parallel.out

simulation_parallel.out: simulation_parallel.cpp
	c++ $(CXX_FLAGS) simulation_parallel.cpp -o simulation_parallel.out

clean:
	$(RM) simulation_parallel.out