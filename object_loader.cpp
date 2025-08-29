// object_loader.cpp
#include "object_loader.h"
#include "simulation.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <cctype>
#include <omp.h>
#include <random>
#include <algorithm>
#include <functional>
#include <thread>
#include <unordered_map>

// 3D point
struct Point {
    float x, y, z;
    Point() : x(0), y(0), z(0) {}
    Point(float x, float y, float z) : x(x), y(y), z(z) {}
};

// Triangle made of 3 points
struct Triangle {
    Point v1, v2, v3;
};

// Simple uniform voxel grid for spatial acceleration
struct VoxelGrid {
    float cellSize;
    Point min;
    int nx, ny, nz;
    std::vector<bool> occupied;

    VoxelGrid() = default;

    // Build grid from triangles
    void init(const Point& grid_min, float cell_size, int cells) {
        min = grid_min;
        cellSize = cell_size;
        nx = cells;
        ny = cells;
        nz = cells;
        occupied.assign(nx * ny * nz, false);
    }

    inline int index(int x, int y, int z) const {
        return x + y * nx + z * nx * ny;
    }

    void insert(const Triangle& tri) {
        // Compute triangle bounds
        Point tmin = tri.v1, tmax = tri.v1;
#define UPDATE(p) \
    tmin.x = std::min(tmin.x, p.x); tmax.x = std::max(tmax.x, p.x); \
    tmin.y = std::min(tmin.y, p.y); tmax.y = std::max(tmax.y, p.y); \
    tmin.z = std::min(tmin.z, p.z); tmax.z = std::max(tmax.z, p.z);
        UPDATE(tri.v2)
        UPDATE(tri.v3)
#undef UPDATE

        // Rasterize into grid
        int ix0 = std::max(0, (int)((tmin.x - min.x) / cellSize));
        int iy0 = std::max(0, (int)((tmin.y - min.y) / cellSize));
        int iz0 = std::max(0, (int)((tmin.z - min.z) / cellSize));
        int ix1 = std::min(nx - 1, (int)((tmax.x - min.x) / cellSize));
        int iy1 = std::min(ny - 1, (int)((tmax.y - min.y) / cellSize));
        int iz1 = std::min(nz - 1, (int)((tmax.z - min.z) / cellSize));

        for (int iz = iz0; iz <= iz1; ++iz)
            for (int iy = iy0; iy <= iy1; ++iy)
                for (int ix = ix0; ix <= ix1; ++ix)
                    occupied[index(ix, iy, iz)] = true;
    }

    bool contains(float x, float y, float z) const {
        if (x < min.x || y < min.y || z < min.z) return false;
        int ix = (x - min.x) / cellSize;
        int iy = (y - min.y) / cellSize;
        int iz = (z - min.z) / cellSize;
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz)
            return false;
        return occupied[index(ix, iy, iz)];
    }
};

// Trim whitespace
std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end = s.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

// Read STL file (ASCII or binary)
std::vector<Triangle> readSTL(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open STL file: " << filename << "\n";
        return {};
    }

    std::string line;
    std::getline(file, line);
    bool isBinary = (trim(line).find("solid") != 0);
    file.close();

    std::vector<Triangle> triangles;

    if (isBinary) {
        file.open(filename, std::ios::binary);
        if (!file) return {};

        file.seekg(80);
        uint32_t numTriangles;
        file.read(reinterpret_cast<char*>(&numTriangles), 4);

        triangles.reserve(numTriangles);
        for (uint32_t i = 0; i < numTriangles; ++i) {
            float nx, ny, nz;
            file.read(reinterpret_cast<char*>(&nx), 4);
            file.read(reinterpret_cast<char*>(&ny), 4);
            file.read(reinterpret_cast<char*>(&nz), 4);

            Triangle tri;
            file.read(reinterpret_cast<char*>(&tri.v1.x), 4);
            file.read(reinterpret_cast<char*>(&tri.v1.y), 4);
            file.read(reinterpret_cast<char*>(&tri.v1.z), 4);
            file.read(reinterpret_cast<char*>(&tri.v2.x), 4);
            file.read(reinterpret_cast<char*>(&tri.v2.y), 4);
            file.read(reinterpret_cast<char*>(&tri.v2.z), 4);
            file.read(reinterpret_cast<char*>(&tri.v3.x), 4);
            file.read(reinterpret_cast<char*>(&tri.v3.y), 4);
            file.read(reinterpret_cast<char*>(&tri.v3.z), 4);

            uint16_t attr;
            file.read(reinterpret_cast<char*>(&attr), 2);

            triangles.push_back(tri);
        }
    } else {
        file.open(filename);
        if (!file) return {};

        Triangle tri;
        int vertexIndex = 0;

        while (std::getline(file, line)) {
            line = trim(line);
            if (line.substr(0, 8) == "facet normal") continue;
            if (line == "outer loop") { vertexIndex = 0; continue; }
            if (line == "endloop") continue;
            if (line == "endfacet") {
                if (vertexIndex == 3) triangles.push_back(tri);
                continue;
            }
            if (line.substr(0, 6) == "vertex") {
                std::istringstream iss(line.substr(6));
                float x, y, z;
                if (iss >> x >> y >> z) {
                    if (vertexIndex == 0) tri.v1 = Point(x, y, z);
                    else if (vertexIndex == 1) tri.v2 = Point(x, y, z);
                    else if (vertexIndex == 2) tri.v3 = Point(x, y, z);
                    vertexIndex = (vertexIndex + 1) % 4;
                }
            }
        }
    }

    std::cout << "Loaded " << triangles.size() << " triangles.\n";
    return triangles;
}

// Degrees to radians
float deg2rad(float deg) {
    return deg * M_PI / 180.0f;
}

// Rotate point around origin using intrinsic ZYX Euler angles
Point rotatePoint(const Point& p, float rot_x_deg, float rot_y_deg, float rot_z_deg) {
    float rx = deg2rad(rot_x_deg);
    float ry = deg2rad(rot_y_deg);
    float rz = deg2rad(rot_z_deg);

    float cx = cosf(rx), sx = sinf(rx);
    float cy = cosf(ry), sy = sinf(ry);
    float cz = cosf(rz), sz = sinf(rz);

    // Combined rotation matrix: R = R_x * R_y * R_z
    float x = p.x;
    float y = p.y;
    float z = p.z;

    Point out;
    out.x = (cy * cz) * x + (-cy * sz) * y + (sy) * z;
    out.y = (sx * sy * cz + cx * sz) * x + (-sx * sy * sz + cx * cz) * y + (-sx * cy) * z;
    out.z = (-cx * sy * cz + sx * sz) * x + (cx * sy * sz + sx * cz) * y + (cx * cy) * z;

    return out;
}

// Fast ray-triangle intersection (Möller–Trumbore)
bool rayIntersectsTriangle(const Point& orig, const Point& dir, const Triangle& tri) {
    const float EPS = 1e-6f;
    Point edge1 = {tri.v2.x - tri.v1.x, tri.v2.y - tri.v1.y, tri.v2.z - tri.v1.z};
    Point edge2 = {tri.v3.x - tri.v1.x, tri.v3.y - tri.v1.y, tri.v3.z - tri.v1.z};

    Point h = {
        dir.y * edge2.z - dir.z * edge2.y,
        dir.z * edge2.x - dir.x * edge2.z,
        dir.x * edge2.y - dir.y * edge2.x
    };
    float a = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;
    if (std::abs(a) < EPS) return false;

    float f = 1.0f / a;
    Point s = {orig.x - tri.v1.x, orig.y - tri.v1.y, orig.z - tri.v1.z};
    float u = f * (s.x * h.x + s.y * h.y + s.z * h.z);
    if (u < 0.0f || u > 1.0f) return false;

    Point q = {
        s.y * edge1.z - s.z * edge1.y,
        s.z * edge1.x - s.x * edge1.z,
        s.x * edge1.y - s.y * edge1.x
    };
    float v = f * (dir.x * q.x + dir.y * q.y + dir.z * q.z);
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);
    return t > 1e-3f;
}

// Fast: single ray with caller-provided direction
bool isInsideMeshSingleRay(const Point& p, const Point& rayDir, const std::vector<Triangle>& triangles) {
    int intersections = 0;
    for (const auto& tri : triangles) {
        if (rayIntersectsTriangle(p, rayDir, tri)) {
            intersections++;
        }
    }
    return (intersections % 2) == 1;
}

// Check if point is inside mesh using multiple rays
bool isInsideMesh(const Point& p, const std::vector<Triangle>& triangles) {
    Point rays[] = {
        {1.0f, 0.1f, 0.2f},
        {0.1f, 1.0f, 0.2f},
        {0.1f, 0.2f, 1.0f},
        {1.0f, 1.0f, 1.0f}
    };

    for (const auto& rayDir : rays) {
        int intersections = 0;
        for (const auto& tri : triangles) {
            if (rayIntersectsTriangle(p, rayDir, tri)) {
                intersections++;
            }
        }
        if ((intersections % 2) == 0) {
            return false;
        }
    }
    return true;
}

// Main function: load, rotate, scale, and add to simulation
void loadSTLIntoObstacles(
    const char* stlFile,
    Simulation& sim,
    float scale,
    float rot_x,
    float rot_y,
    float rot_z,
    float translate_x,
    float translate_y,
    float translate_z
) {
    auto triangles = readSTL(stlFile);
    if (triangles.empty()) {
        std::cerr << "Failed to load STL: " << stlFile << "\n";
        return;
    }

    // Compute original bounding box
    Point orig_min = {1e6f, 1e6f, 1e6f};
    Point orig_max = {-1e6f, -1e6f, -1e6f};

    // Step 1: Compute object center
    Point objCenter = {
        (orig_min.x + orig_max.x) * 0.5f,
        (orig_min.y + orig_max.y) * 0.5f,
        (orig_min.z + orig_max.z) * 0.5f
    };

    // Rotate all triangles in parallel
    std::vector<Triangle> rotatedTris;
    rotatedTris.resize(triangles.size()); // Pre-allocate for thread safety

    #pragma omp parallel for
    for (size_t i = 0; i < triangles.size(); ++i) {
        const Triangle& tri = triangles[i];

        // Helper: rotate a point around objCenter
        auto rotateVertex = [&](const Point& p) -> Point {
            Point local = {p.x - objCenter.x, p.y - objCenter.y, p.z - objCenter.z};
            Point r = rotatePoint(local, rot_x, rot_y, rot_z);
            return {r.x + objCenter.x, r.y + objCenter.y, r.z + objCenter.z};
        };

        rotatedTris[i].v1 = rotateVertex(tri.v1);
        rotatedTris[i].v2 = rotateVertex(tri.v2);
        rotatedTris[i].v3 = rotateVertex(tri.v3);
    }

    // Step 2: Find maximum distance from center to any vertex → bounding sphere radius
    float maxRadius = 0.0f;

    auto distance_sq = [](const Point& a, const Point& b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float dz = a.z - b.z;
        return dx*dx + dy*dy + dz*dz;
    };

    for (const auto& tri : triangles) {
        float d1 = distance_sq(tri.v1, objCenter);
        float d2 = distance_sq(tri.v2, objCenter);
        float d3 = distance_sq(tri.v3, objCenter);
        maxRadius = std::max({maxRadius, d1, d2, d3});
    }
    maxRadius = std::sqrt(maxRadius);

    // Step 3: Bounding box is a cube of side = 2 * maxRadius, centered at objCenter
    Point safe_min = {
        objCenter.x - maxRadius,
        objCenter.y - maxRadius,
        objCenter.z - maxRadius
    };
    Point safe_max = {
        objCenter.x + maxRadius,
        objCenter.y + maxRadius,
        objCenter.z + maxRadius
    };

    // Step 4: Add extra padding in world space (e.g., 5% of radius)
    float extra_pad = maxRadius * 0.05f;
    Point padded_min = {
        safe_min.x - extra_pad,
        safe_min.y - extra_pad,
        safe_min.z - extra_pad
    };
    Point padded_max = {
        safe_max.x + extra_pad,
        safe_max.y + extra_pad,
        safe_max.z + extra_pad
    };

    // === Voxel grid setup ===
    float objSize = std::max({
        padded_max.x - padded_min.x,
        padded_max.y - padded_min.y,
        padded_max.z - padded_min.z
    });

    float resolution = std::max(objSize / 200.0f, 0.02f);  // finer resolution

    int nx = (int)((padded_max.x - padded_min.x) / resolution);
    int ny = (int)((padded_max.y - padded_min.y) / resolution);
    int nz = (int)((padded_max.z - padded_min.z) / resolution);

    std::cout << "Rotated object voxelization:\n";
    std::cout << "  Grid: " << nx << " x " << ny << " x " << nz << "\n";
    std::cout << "  Resolution: " << resolution << "\n";
    std::cout << "  Rotations: X=" << rot_x << "° Y=" << rot_y << "° Z=" << rot_z << "°\n";

    // === Build spatial acceleration structure ===
    VoxelGrid voxelGrid;
    float coarseRes = resolution * 5.0f;  // 5x voxel size
    int gridSize = 64;  // Adjust: 64 → good for most objects

    voxelGrid.init(padded_min, coarseRes, gridSize);

    #pragma omp parallel for
    for (size_t i = 0; i < rotatedTris.size(); ++i) {
        voxelGrid.insert(rotatedTris[i]);
    }

    std::cout << "Built spatial grid for fast rejection.\n";

    // === Voxelization with early rejection and single ray ===
    int added = 0;

    #pragma omp parallel reduction(+:added)
    {
        // Thread-local RNG
        std::minstd_rand gen(static_cast<unsigned int>(std::hash<std::thread::id>{}(std::this_thread::get_id())));
        std::uniform_real_distribution<float> unit(0.1f, 1.0f);  // avoid zero

        #pragma omp for collapse(3)
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    Point p;
                    p.x = padded_min.x + i * resolution;
                    p.y = padded_min.y + j * resolution;
                    p.z = padded_min.z + k * resolution;

                    // Early rejection: skip if not near any triangle
                    if (!voxelGrid.contains(p.x, p.y, p.z)) {
                        continue;
                    }

                    // Add small jitter to avoid alignment
                    p.x += (gen() % 1000) * 1e-6f - 5e-4f;
                    p.y += (gen() % 1000) * 1e-6f - 5e-4f;
                    p.z += (gen() % 1000) * 1e-6f - 5e-4f;

                    // Random ray direction (unit-like)
                    Point rayDir = {unit(gen), unit(gen), unit(gen)};

                    if (isInsideMeshSingleRay(p, rayDir, rotatedTris)) {
                        // Map to simulation grid
                        float objSize = std::max({padded_max.x - padded_min.x,
                                                padded_max.y - padded_min.y,
                                                padded_max.z - padded_min.z});
                        float gridScale = scale * std::min({(float)sim.width, (float)sim.height, (float)sim.depth}) / objSize;
                        Point gridCenter = {(float)sim.width/2, (float)sim.height/2, (float)sim.depth/2};

                        int gx = (int)((p.x - objCenter.x) * gridScale + gridCenter.x + translate_x);
                        int gy = (int)((p.y - objCenter.y) * gridScale + gridCenter.y + translate_y);
                        int gz = (int)((p.z - objCenter.z) * gridScale + gridCenter.z + translate_z);

                        if (gx >= 1 && gx <= sim.width &&
                            gy >= 1 && gy <= sim.height &&
                            gz >= 1 && gz <= sim.depth) {
                            #pragma omp critical
                            {
                                sim.addObstacle(gx, gy, gz);
                            }
                            added++;
                        }
                    }
                }
            }
        }
    }

    std::cout << "Added " << added << " obstacle points.\n";
}