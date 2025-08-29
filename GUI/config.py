"""
Global configuration for the fluid simulation visualizer.
This must match the C++ simulation parameters.
"""

# The C++ code stores the *padded* grid (width+2)*(height+2)*(depth+2)
# per time step, so we include the padding here as well.
scale = 1
width  = (128 * scale) + 2          # X-size (including the two walls)
height = (64 * scale) + 2           # Y-size (including the two walls)
depth  = (64 * scale) + 2           # Z-size (including the two walls)

# Custom colour map for density (white → green → blue → red)
from matplotlib.colors import LinearSegmentedColormap
density_cmap = LinearSegmentedColormap.from_list(
    "density_cmap",
    ["white", "lightgreen", "green", "deepskyblue", "blue", "darkred", "red"],
)

# Streamline parameters
STREAMLINE_DENSITY = 30   # Number of seed points per dimension
STREAMLINE_PROXIMITY = 2 # Proximity to obstacle for streamlines to be visible
INTEGRATION_STEPS = 100    # Steps to integrate each streamline
INTEGRATION_STEP_SIZE = 0.2  # Step size for integration
VELOCITY_CHANGE_THRESHOLD = 0.1  # Only show streamlines where velocity changes significantly