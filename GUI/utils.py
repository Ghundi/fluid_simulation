"""
Utility functions for the fluid simulation visualizer.
"""

import numpy as np
from skimage import measure

import config

def generate_obstacle_mesh(obs_data):
    """
    Generate vertices, faces, and colors for a 3D mesh representing obstacles.
    Uses marching cubes algorithm to create a proper surface.
    """
    # Extract the surface using marching cubes
    try:
        verts, faces, _, _ = measure.marching_cubes(obs_data, level=0.5)
        
        # Solid gray color with full opacity
        colors = np.ones((verts.shape[0], 4))
        colors[:, 0] = 0.5  # R
        colors[:, 1] = 0.5  # G
        colors[:, 2] = 0.5  # B
        colors[:, 3] = 1.0  # Alpha (fully opaque)
        
        mesh_data = {
            'vertexes': verts,
            'faces': faces,
            'vertex_colors': colors
        }
        return mesh_data
    except:
        # If no obstacles, return empty mesh
        return {
            'vertexes': np.array([]),
            'faces': np.array([]),
            'vertex_colors': np.array([])
        }

def _interpolate_scalar(grid, x, y, z):
    """Performs trilinear interpolation on a 3D grid."""
    # Ensure coordinates are within bounds
    x = np.clip(x, 0, grid.shape[0] - 1.001)
    y = np.clip(y, 0, grid.shape[1] - 1.001)
    z = np.clip(z, 0, grid.shape[2] - 1.001)

    x0, y0, z0 = int(x), int(y), int(z)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    # Fractional parts
    xd, yd, zd = x - x0, y - y0, z - z0

    # Get values at the 8 corners of the cube
    c000 = grid[x0, y0, z0]
    c100 = grid[x1, y0, z0]
    c010 = grid[x0, y1, z0]
    c110 = grid[x1, y1, z0]
    c001 = grid[x0, y0, z1]
    c101 = grid[x1, y0, z1]
    c011 = grid[x0, y1, z1]
    c111 = grid[x1, y1, z1]

    # Interpolate along x
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    # Interpolate along y
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # Interpolate along z
    return c0 * (1 - zd) + c1 * zd

def _interpolate_vector(vx, vy, vz, x, y, z):
    """Interpolates a 3D vector field."""
    u = _interpolate_scalar(vx, x, y, z)
    v = _interpolate_scalar(vy, x, y, z)
    w = _interpolate_scalar(vz, x, y, z)
    return np.array([u, v, w])

def _integrate_streamline_part(start_pos, vx, vy, vz, obs_data, max_steps, direction):
    """Integrates a streamline in one direction (forward or backward)."""
    streamline_points = [start_pos]
    streamline_velocities = [_interpolate_vector(vx, vy, vz, *start_pos)]
    
    pos = start_pos.copy()

    for _ in range(max_steps):
        vec = _interpolate_vector(vx, vy, vz, pos[0], pos[1], pos[2])
        speed = np.linalg.norm(vec)

        if speed < 1e-6: break
        
        # Move one step
        pos += direction * (vec / speed) * config.INTEGRATION_STEP_SIZE

        # Check for boundaries or obstacles
        if not (1 <= pos[0] < config.width - 1 and
                1 <= pos[1] < config.height - 1 and
                1 <= pos[2] < config.depth - 1):
            break
        if _interpolate_scalar(obs_data, pos[0], pos[1], pos[2]) > 0.5:
            break

        streamline_points.append(pos.copy())
        streamline_velocities.append(vec)

    return streamline_points, streamline_velocities


def generate_streamlines(vx, vy, vz, obs_data, max_length=config.INTEGRATION_STEPS):
    """
    Generate streamlines from the velocity field, but only keep those near obstacles.
    This corrected version uses trilinear interpolation and efficient obstacle proximity filtering.
    """
    streamlines = []
    streamline_colors = []
    
    # Precompute obstacle bounding box for fast culling (critical optimization)
    obstacle_indices = np.where(obs_data > 0.5)
    if len(obstacle_indices[0]) > 0:
        min_x, max_x = np.min(obstacle_indices[0]), np.max(obstacle_indices[0])
        min_y, max_y = np.min(obstacle_indices[1]), np.max(obstacle_indices[1])
        min_z, max_z = np.min(obstacle_indices[2]), np.max(obstacle_indices[2])
        obstacle_min = np.array([min_x, min_y, min_z]) - (config.STREAMLINE_PROXIMITY / 10)
        obstacle_max = np.array([max_x, max_y, max_z]) + (config.STREAMLINE_PROXIMITY / 10)
    else:
        # No obstacles, return empty streamlines
        return [], []
    
    # Define Seed Points
    x_seeds = np.linspace(1, config.width - 2, config.STREAMLINE_DENSITY)
    y_seeds = np.linspace(1, config.height - 2, config.STREAMLINE_DENSITY // 2)
    z_seeds = np.linspace(1, config.depth - 2, config.STREAMLINE_DENSITY // 2)
    
    # Process each seed point
    for z_seed in z_seeds:
        for y_seed in y_seeds:
            for x_seed in x_seeds:
                start_pos = np.array([x_seed, y_seed, z_seed])
                
                # FAST CULLING: Skip seeds far from obstacle bounding box
                if (start_pos[0] < obstacle_min[0] or start_pos[0] > obstacle_max[0] or
                    start_pos[1] < obstacle_min[1] or start_pos[1] > obstacle_max[1] or
                    start_pos[2] < obstacle_min[2] or start_pos[2] > obstacle_max[2]):
                    continue
                
                # Skip seeds inside obstacles
                if obs_data[int(x_seed), int(y_seed), int(z_seed)] > 0.5:
                    continue
                
                # Integrate backward from the seed
                backward_points, backward_velocities = _integrate_streamline_part(
                    start_pos, vx, vy, vz, obs_data, max_length // 2, direction=-1.0
                )
                
                # Integrate forward from the seed
                forward_points, forward_velocities = _integrate_streamline_part(
                    start_pos, vx, vy, vz, obs_data, max_length // 2, direction=1.0
                )
                
                # Combine backward and forward parts
                full_streamline = backward_points[::-1][:-1] + forward_points
                full_velocities = backward_velocities[::-1][:-1] + forward_velocities
                
                # Filter short streamlines
                if len(full_streamline) <= 5:
                    continue
                
                # Filter by velocity change
                max_change = 0.0
                for i in range(1, len(full_velocities)):
                    change = np.linalg.norm(full_velocities[i] - full_velocities[i-1])
                    if change > max_change:
                        max_change = change
                if max_change < config.VELOCITY_CHANGE_THRESHOLD:
                    continue
                
                # CRITICAL FILTER: Only keep streamlines near obstacles
                streamline_near_obstacle = False
                # Check a subset of points for efficiency (every 3rd point)
                for i in range(0, len(full_streamline), 3):
                    pt = full_streamline[i]
                    # Check if point is within expanded obstacle bounding box
                    if (obstacle_min[0] <= pt[0] <= obstacle_max[0] and
                        obstacle_min[1] <= pt[1] <= obstacle_max[1] and
                        obstacle_min[2] <= pt[2] <= obstacle_max[2]):
                        streamline_near_obstacle = True
                        break
                
                if not streamline_near_obstacle:
                    continue
                
                # Color the streamline based on speed
                speeds = [np.linalg.norm(v) for v in full_velocities]
                max_speed = max(speeds) if speeds else 0.0
                
                # Normalize speed for colormap
                norm_speed = max_speed / (np.max([vx, vy, vz]) + 1e-6)
                
                colour = np.array(config.density_cmap(min(norm_speed, 1.0)))
                streamline_colors.append(colour)
                
                # Store the accepted streamline
                streamlines.append(np.array(full_streamline))
    
    return streamlines, streamline_colors