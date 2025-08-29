# Makefile

# Compiler and flags
CXX = c++
CXX_FLAGS = -std=c++20 -O2 -Wall -Wextra -Wpedantic -fopenmp

# Output executable
TARGET = simulation.out

# Source files
SOURCES = simulation.cpp object_loader.cpp

# Object files (auto-generated from sources)
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Link object files into executable
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(CXX_FLAGS)

# Compile .cpp files to .o
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Clean build files
clean:
	$(RM) $(OBJECTS) $(TARGET)

# Optional: rebuild everything
rebuild: clean all

.PHONY: all clean rebuild
