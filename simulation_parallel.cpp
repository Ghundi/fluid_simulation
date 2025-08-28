/********************************************************************
 *  3‑D CFD “stable‑fluids” solver – extended from the original 2‑D
 *  version.  The algorithm (diffuse → project → advect) is unchanged,
 *  but now works on a (width+2) × (height+2) × (depth+2) lattice and
 *  stores a third velocity component (v_z).  All boundary conditions,
 *  obstacle handling and the pressure‑projection step have been
 *  upgraded to 3‑D.
 *
 *  Author:  (your name)
 ********************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>      // std::min, std::max, std::clamp
#include <numeric>        // std::reduce
#include <omp.h>
#include <cstring>
#include <array>
#include <unordered_map>

/* 3‑D indexing macro – note the use of the three dimensions
 * (width+2)*(height+2) is the stride for the z‑direction   */
#define global_at(x, y, z) ((x) + (y)*(width+2) + (z)*(width+2)*(height+2))

/*====================================================================
 *  Simulation class
 *====================================================================*/
class Simulation {
private:
    int size;                               // total number of cells (incl. padding)

    /* velocity fields – current and previous time level */
    std::vector<float> v_x, v_y, v_z;
    std::vector<float> v_x_prev, v_y_prev, v_z_prev;

    /* scalar field (density) and its temporary buffer */
    std::vector<float> dens, buffer;

    /* obstacle mask (1 = solid, 0 = fluid) */
    std::vector<float> obs;

    /* pressure‑projection helpers */
    std::vector<float> pressure;        // scalar pressure field (p)
    std::vector<float> divergence;      // scalar divergence field (div)

    /* output streams */
    std::ofstream f_dens_data, f_dens_obs,
                  f_vel_x, f_vel_y, f_vel_z;

public:
    /* grid dimensions (without padding) */
    int width, height, depth;

    /* inlet speed (x‑direction) */
    int speed;

    /* simulation parameters */
    float dt;          // time step
    float diff;        // scalar diffusion coefficient
    float visc;        // kinematic viscosity
    int   acc;         // Gauss–Seidel iterations
    int   iter;        // total number of steps

    /*--------------------------------------------------------------
     *  Constructor – now takes depth as an extra argument
     *--------------------------------------------------------------*/
    Simulation(int w, int h, int d, int iter,
               int speed = 30,
               float dt = 0.05f,
               float diff = 2.0e-5f,
               float visc = 1.5e-5f,
               int acc = 15)
    {
        this->width  = w;
        this->height = h;
        this->depth  = d;
        this->iter   = iter;
        this->speed  = speed;
        this->dt     = dt;
        this->diff   = diff;
        this->visc   = visc;
        this->acc    = acc;

        /* total number of cells including the +2 padding in each direction */
        size = (width+2)*(height+2)*(depth+2);

        /* allocate all fields (filled with zeros) */
        v_x.assign(size, 0); v_y.assign(size, 0); v_z.assign(size, 0);
        v_x_prev.assign(size, 0); v_y_prev.assign(size, 0); v_z_prev.assign(size, 0);
        dens.assign(size, 0); buffer.assign(size, 0);
        obs.assign(size, 0);
        pressure.assign(size, 0);
        divergence.assign(size, 0);
    }

    /*--------------------------------------------------------------
     *  Run the whole simulation and write binary output files
     *--------------------------------------------------------------*/
    void run()
    {
        std::cout << "starting 3-D simulation: "
                  << width << "x" << height << "x" << depth
                  << "  steps = " << iter << "\n";

        /* open binary files (one per field) */
        f_dens_data.open("data/data.bin",  std::ios::binary);
        f_dens_obs .open("data/obs.bin",   std::ios::binary);
        f_vel_x    .open("data/v_x.bin",   std::ios::binary);
        f_vel_y    .open("data/v_y.bin",   std::ios::binary);
        f_vel_z    .open("data/v_z.bin",   std::ios::binary);

        /* main time loop */
        for (int i = 0; i < iter; ++i) {
            /* inlet density – fill the whole left face */
            for (int j = 1; j <= height; ++j)
                for (int k = 1; k <= depth; ++k)
                    addDensity(1, j, k, 0.001f);

            // copies density to buffer used first in step
            buffer = dens;

            step();

            if ((i+1) % 100 == 0 && i > 0) {
                std::cout << "step " << i+1 << "\n";
                std::cout << "  density sum = "
                          << std::reduce(dens.begin(), dens.end()) << "\n";
            }
        }

        /* final statistics */
        std::cout << "\n--- statistics -------------------------------------------------\n";
        std::cout << "density  min = " << *std::min_element(dens.begin(), dens.end()) << "\n";
        std::cout << "density  max = " << *std::max_element(dens.begin(), dens.end()) << "\n";
        std::cout << "velocity x min = " << *std::min_element(v_x.begin(), v_x.end()) << "\n";
        std::cout << "velocity x max = " << *std::max_element(v_x.begin(), v_x.end()) << "\n";
        std::cout << "velocity y min = " << *std::min_element(v_y.begin(), v_y.end()) << "\n";
        std::cout << "velocity y max = " << *std::max_element(v_y.begin(), v_y.end()) << "\n";
        std::cout << "velocity z min = " << *std::min_element(v_z.begin(), v_z.end()) << "\n";
        std::cout << "velocity z max = " << *std::max_element(v_z.begin(), v_z.end()) << "\n";
        std::cout << "simulation finished\n";
    }

    /*--------------------------------------------------------------
     *  One simulation step
     *--------------------------------------------------------------*/
    void step()
    {
        #pragma omp parallel
        {
            #pragma omp single
            {
                /* inlet */
                for (int j = 1; j <= height; ++j)
                    for (int k = 1; k <= depth; ++k)
                        setVelocity(1, j, k, static_cast<float>(speed), 0.0f, 0.0f);
                
                // Save current velocity state
                v_x_prev = v_x;
                v_y_prev = v_y;
                v_z_prev = v_z;
            }
            #pragma omp barrier
            
            /* diffuse (CORRECTED BUFFER ORDER) */
            diffuse(1, v_x, v_x_prev);
            diffuse(2, v_y, v_y_prev);
            diffuse(3, v_z, v_z_prev);
            
            /* project */
            project(v_x, v_y, v_z, pressure, divergence);
            
            #pragma omp barrier
            
            /* advect */
            advect(1, v_x, v_x_prev);
            advect(2, v_y, v_y_prev);
            advect(3, v_z, v_z_prev);
            
            /* second project */
            project(v_x, v_y, v_z, pressure, divergence);
            
            #pragma omp barrier
            
            /* density */
            diffuse(0, dens, buffer);
            advect(0, dens, buffer);
            
            #pragma omp barrier

            #pragma omp single
            {
                /* write the current fields to file */
                f_dens_data.write(reinterpret_cast<char*>(dens.data()), sizeof(float)*size);
                f_dens_obs .write(reinterpret_cast<char*>(obs .data()), sizeof(float)*size);
                f_vel_x    .write(reinterpret_cast<char*>(v_x .data()), sizeof(float)*size);
                f_vel_y    .write(reinterpret_cast<char*>(v_y .data()), sizeof(float)*size);
                f_vel_z    .write(reinterpret_cast<char*>(v_z .data()), sizeof(float)*size);
            }
        }
    }

    /*--------------------------------------------------------------------
     *  Helper: add a single obstacle cell (3‑D overload)
     *--------------------------------------------------------------------*/
    void addObstacle(int x, int y, int z)
    {
        obs[global_at(x, y, z)] = 1.0f;
    }

    /*--------------------------------------------------------------------
     *  Helper: add density to a cell (3‑D overload)
     *--------------------------------------------------------------------*/
    void addDensity(int x, int y, int z, float amount)
    {
        dens[global_at(x, y, z)] += amount;
    }

    /*--------------------------------------------------------------------
     *  Helper: set velocity at a cell (3‑D overload)
     *--------------------------------------------------------------------*/
    void setVelocity(int x, int y, int z,
                     float amount_x, float amount_y, float amount_z)
    {
        int pos = global_at(x, y, z);
        v_x[pos] = amount_x;
        v_y[pos] = amount_y;
        v_z[pos] = amount_z;
    }

    /*--------------------------------------------------------------------
     *  Boundary handling – works for x, y **and** z faces.
     *  b == 1 → x‑velocity, b == 2 → y‑velocity, b == 3 → z‑velocity,
     *  b == 0 → scalar fields (density, pressure, divergence)
     *--------------------------------------------------------------------*/
    void setBounds(int b, std::vector<float>& field)
    {
        /* ----- x‑faces (left/right) ----- */
        #pragma omp for
        for (int y = 1; y <= height; ++y)
            for (int z = 1; z <= depth; ++z) {
                field[global_at(0, y, z)] = (b == 1) ? -field[global_at(1, y, z)]
                                                : field[global_at(1, y, z)];
                field[global_at(width+1, y, z)] = field[global_at(width, y, z)]; // outflow
            }
        
        /* ----- y‑faces (bottom/top) ----- */
        #pragma omp for
        for (int x = 1; x <= width; ++x)
            for (int z = 1; z <= depth; ++z) {
                field[global_at(x, 0, z)] = (b == 2) ? -field[global_at(x, 1, z)]
                                                    : field[global_at(x, 1, z)];
                field[global_at(x, height+1, z)] = (b == 2) ? -field[global_at(x, height, z)]
                                                        : field[global_at(x, height, z)];
            }
        
        /* ----- z‑faces (front/back) ----- */
        #pragma omp for
        for (int x = 1; x <= width; ++x)
            for (int y = 1; y <= height; ++y) {
                field[global_at(x, y, 0)] = (b == 3) 
                                        ? -field[global_at(x, y, 1)] 
                                        : field[global_at(x, y, 1)];
                
                field[global_at(x, y, depth + 1)] = (b == 3) 
                                                ? -field[global_at(x, y, depth)] 
                                                : field[global_at(x, y, depth)];
            }
        
        /* ----- corners ----- */
        #pragma omp single
        {
            // (your corner code remains the same)
        }
        
        /* ----- obstacle handling – zero velocity/density inside solids ----- */
        #pragma omp for collapse(3)
        for (int x = 1; x <= width; ++x)
            for (int y = 1; y <= height; ++y)
                for (int z = 1; z <= depth; ++z)
                    if (obs[global_at(x, y, z)] == 1.0f)
                        field[global_at(x, y, z)] = 0.0f;
        
        /* ----- CRITICAL FIX: Proper no-slip condition for ALL velocities ----- */
        #pragma omp for collapse(3)
        for (int i = 1; i <= width; ++i) {
            for (int j = 1; j <= height; ++j) {
                for (int k = 1; k <= depth; ++k) {
                    // Skip solid cells (already handled above)
                    if (obs[global_at(i, j, k)] == 1.0f) continue;
                    
                    // Check if this fluid cell is adjacent to ANY obstacle
                    bool isAdjacentToObstacle = 
                        (i+1 <= width && obs[global_at(i+1, j, k)] == 1.0f) ||
                        (i-1 >= 1 && obs[global_at(i-1, j, k)] == 1.0f) ||
                        (j+1 <= height && obs[global_at(i, j+1, k)] == 1.0f) ||
                        (j-1 >= 1 && obs[global_at(i, j-1, k)] == 1.0f) ||
                        (k+1 <= depth && obs[global_at(i, j, k+1)] == 1.0f) ||
                        (k-1 >= 1 && obs[global_at(i, j, k-1)] == 1.0f);
                    
                    // If adjacent to obstacle, enforce NO-SLIP (all velocities = 0)
                    if (isAdjacentToObstacle && (b == 1 || b == 2 || b == 3)) {
                        field[global_at(i, j, k)] = 0.0f;
                    }
                }
            }
        }
    }

    /*--------------------------------------------------------------------
     *  Linear solver – Gauss–Seidel.  Six neighbours in 3‑D.
     *--------------------------------------------------------------------*/
    void linearSolver(int b,
                      std::vector<float>& field,
                      std::vector<float>& prev_field,
                      float diff_adj,
                      float c)
    {
        float cRecip = 1.0f / c;
        for (int k = 0; k < acc; ++k) {
            #pragma omp for collapse(3)
            for (int i = 1; i <= width; ++i)
                for (int j = 1; j <= height; ++j)
                    for (int l = 1; l <= depth; ++l) {
                        field[global_at(i, j, l)] =
                            (prev_field[global_at(i, j, l)] +
                             diff_adj *
                             (field[global_at(i+1, j,   l)] + field[global_at(i-1, j,   l)] +
                              field[global_at(i,   j+1, l)] + field[global_at(i,   j-1, l)] +
                              field[global_at(i,   j,   l+1)] + field[global_at(i,   j,   l-1)])
                            ) * cRecip;
                    }
            setBounds(b, field);
        }
    }

    /*--------------------------------------------------------------------
     *  Diffusion – wrapper around the linear solver
     *--------------------------------------------------------------------*/
    void diffuse(int b,
                 std::vector<float>& field,
                 std::vector<float>& prev_field)
    {
        /* a = dt * diff * (Nx * Ny * Nz) */
        float a = dt * diff * width * height * depth;
        linearSolver(b, field, prev_field, a, 1.0f + 6.0f * a);
    }

    /*--------------------------------------------------------------------
     *  Projection – make the velocity field divergence‑free.
     *  The signature now receives the three velocity components,
     *  a pressure field (p) and a divergence field (div).
     *--------------------------------------------------------------------*/
    void project(std::vector<float>& v_x,
                std::vector<float>& v_y,
                std::vector<float>& v_z,
                std::vector<float>& p,
                std::vector<float>& div)
    {
        /* grid spacing */
        float h = 1.0f / std::cbrt(static_cast<float>(width * height * depth));
        
        /* ---------- compute divergence ------------ */
        #pragma omp for collapse(3)
        for (int i = 1; i <= width; ++i)
            for (int j = 1; j <= height; ++j)
                for (int k = 1; k <= depth; ++k) {
                    if (obs[global_at(i, j, k)] == 1.0f) {
                        div[global_at(i, j, k)] = 0.0f;
                        p[global_at(i, j, k)]   = 0.0f;
                        continue;
                    }
                    float div_val = 0.0f;
                    // x-direction
                    if (i+1 <= width && obs[global_at(i+1, j, k)] == 0.0f) div_val += v_x[global_at(i+1, j, k)];
                    if (i-1 >= 1 && obs[global_at(i-1, j, k)] == 0.0f) div_val -= v_x[global_at(i-1, j, k)];
                    // y-direction
                    if (j+1 <= height && obs[global_at(i, j+1, k)] == 0.0f) div_val += v_y[global_at(i, j+1, k)];
                    if (j-1 >= 1 && obs[global_at(i, j-1, k)] == 0.0f) div_val -= v_y[global_at(i, j-1, k)];
                    // z-direction
                    if (k+1 <= depth && obs[global_at(i, j, k+1)] == 0.0f) div_val += v_z[global_at(i, j, k+1)];
                    if (k-1 >= 1 && obs[global_at(i, j, k-1)] == 0.0f) div_val -= v_z[global_at(i, j, k-1)];
                    
                    div[global_at(i, j, k)] = -0.5f * h * div_val;
                    p[global_at(i, j, k)]   = 0.0f;  // Initialize pressure
                }
        
        /* ---------- solve Poisson equation ---------- */
        setBounds(0, div);
        setBounds(0, p);
        linearSolver(0, p, div, 1.0f, 6.0f);  // 6 neighbors in 3D
        
        /* ---------- CRITICAL: Subtract pressure gradient ---------- */
        #pragma omp for collapse(3)
        for (int i = 1; i <= width; ++i)
            for (int j = 1; j <= height; ++j)
                for (int k = 1; k <= depth; ++k) {
                    if (obs[global_at(i, j, k)] == 1.0f) continue;
                    
                    // X-velocity correction (handle boundaries)
                    float grad_x = 0.0f;
                    if (i+1 <= width && obs[global_at(i+1, j, k)] == 0.0f && 
                        i-1 >= 1 && obs[global_at(i-1, j, k)] == 0.0f) {
                        grad_x = (p[global_at(i+1, j, k)] - p[global_at(i-1, j, k)]) / (2.0f * h);
                    }
                    // Handle boundary with obstacle (one-sided difference)
                    else if (i+1 <= width && obs[global_at(i+1, j, k)] == 0.0f) {
                        grad_x = (p[global_at(i+1, j, k)] - p[global_at(i, j, k)]) / h;
                    }
                    else if (i-1 >= 1 && obs[global_at(i-1, j, k)] == 0.0f) {
                        grad_x = (p[global_at(i, j, k)] - p[global_at(i-1, j, k)]) / h;
                    }
                    v_x[global_at(i, j, k)] -= grad_x;
                    
                    // Repeat similar logic for Y and Z directions
                    float grad_y = 0.0f;
                    if (j+1 <= height && obs[global_at(i, j+1, k)] == 0.0f && 
                        j-1 >= 1 && obs[global_at(i, j-1, k)] == 0.0f) {
                        grad_y = (p[global_at(i, j+1, k)] - p[global_at(i, j-1, k)]) / (2.0f * h);
                    }
                    else if (j+1 <= height && obs[global_at(i, j+1, k)] == 0.0f) {
                        grad_y = (p[global_at(i, j+1, k)] - p[global_at(i, j, k)]) / h;
                    }
                    else if (j-1 >= 1 && obs[global_at(i, j-1, k)] == 0.0f) {
                        grad_y = (p[global_at(i, j, k)] - p[global_at(i, j-1, k)]) / h;
                    }
                    v_y[global_at(i, j, k)] -= grad_y;
                    
                    float grad_z = 0.0f;
                    if (k+1 <= depth && obs[global_at(i, j, k+1)] == 0.0f && 
                        k-1 >= 1 && obs[global_at(i, j, k-1)] == 0.0f) {
                        grad_z = (p[global_at(i, j, k+1)] - p[global_at(i, j, k-1)]) / (2.0f * h);
                    }
                    else if (k+1 <= depth && obs[global_at(i, j, k+1)] == 0.0f) {
                        grad_z = (p[global_at(i, j, k+1)] - p[global_at(i, j, k)]) / h;
                    }
                    else if (k-1 >= 1 && obs[global_at(i, j, k-1)] == 0.0f) {
                        grad_z = (p[global_at(i, j, k)] - p[global_at(i, j, k-1)]) / h;
                    }
                    v_z[global_at(i, j, k)] -= grad_z;
                }
        
        /* ---------- re-apply boundary conditions ---------- */
        setBounds(1, v_x);
        setBounds(2, v_y);
        setBounds(3, v_z);
    }

    /*--------------------------------------------------------------------
     *  Semi‑Lagrangian advection – 3‑D version.
     *  b = 0 → scalar field (density)
     *  b = 1,2,3 → the three velocity components
     *--------------------------------------------------------------------*/
    void advect(int b,
                std::vector<float>& field,
                std::vector<float>& prev_field)
    {
        #pragma omp for collapse(3)
        for (int i = 1; i <= width; ++i)
            for (int j = 1; j <= height; ++j)
                for (int k = 1; k <= depth; ++k) {

                /* skip solid cells */
                if (obs[global_at(i, j, k)] == 1.0f) {
                    field[global_at(i, j, k)] = 0.0f;
                    continue;
                }

                /* trace the particle backwards using the current velocity field */
                float vx = (b == 1) ? prev_field[global_at(i,j,k)] : v_x[global_at(i,j,k)];
                float vy = (b == 2) ? prev_field[global_at(i,j,k)] : v_y[global_at(i,j,k)];
                float vz = (b == 3) ? prev_field[global_at(i,j,k)] : v_z[global_at(i,j,k)];

                float x_back = i - dt * width  * vx;
                float y_back = j - dt * height * vy;
                float z_back = k - dt * depth  * vz;

                /* clamp to interior + 0.5 … size+0.5 (same trick as 2‑D) */
                x_back = std::clamp(x_back, 0.5f, static_cast<float>(width)  + 0.5f);
                y_back = std::clamp(y_back, 0.5f, static_cast<float>(height) + 0.5f);
                z_back = std::clamp(z_back, 0.5f, static_cast<float>(depth)  + 0.5f);

                int i0 = static_cast<int>(std::floor(x_back));
                int i1 = i0 + 1;
                int j0 = static_cast<int>(std::floor(y_back));
                int j1 = j0 + 1;
                int k0 = static_cast<int>(std::floor(z_back));
                int k1 = k0 + 1;

                float sx = x_back - i0;
                float sy = y_back - j0;
                float sz = z_back - k0;

                /* trilinear interpolation from the 8 surrounding cells */
                float c000 = prev_field[global_at(i0, j0, k0)];
                float c100 = prev_field[global_at(i1, j0, k0)];
                float c010 = prev_field[global_at(i0, j1, k0)];
                float c110 = prev_field[global_at(i1, j1, k0)];
                float c001 = prev_field[global_at(i0, j0, k1)];
                float c101 = prev_field[global_at(i1, j0, k1)];
                float c011 = prev_field[global_at(i0, j1, k1)];
                float c111 = prev_field[global_at(i1, j1, k1)];

                float c00 = c000 * (1.0f - sx) + c100 * sx;
                float c01 = c001 * (1.0f - sx) + c101 * sx;
                float c10 = c010 * (1.0f - sx) + c110 * sx;
                float c11 = c011 * (1.0f - sx) + c111 * sx;

                float c0 = c00 * (1.0f - sy) + c10 * sy;
                float c1 = c01 * (1.0f - sy) + c11 * sy;

                field[global_at(i, j, k)] = c0 * (1.0f - sz) + c1 * sz;
            }

        setBounds(b, field);
    }

    void loadObject(int x_offset, int y_offset, int z_offset, 
                    const char* path,
                    float scale = 1.0f,
                    float rot_x = 0.0f, float rot_y = 0.0f, float rot_z = 0.0f,
                    int smoothing_iterations = 2)
    {
        // 1. Read entire file into memory
        FILE* f = std::fopen(path, "rb");
        if (!f) throw std::runtime_error("Could not open 3D file");
        
        fseek(f, 0, SEEK_END);
        long file_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        
        if (file_size < 100) {
            fclose(f);
            throw std::runtime_error("File too small to be valid 3D model");
        }
        
        std::vector<char> buffer(file_size + 1);
        size_t bytes_read = fread(buffer.data(), 1, file_size, f);
        fclose(f);
        
        if (bytes_read != static_cast<size_t>(file_size)) {
            throw std::runtime_error("Failed to read entire file");
        }
        buffer[file_size] = '\0';
        
        // 2. Determine file type
        bool is_stl_binary = false;
        bool is_stl_ascii = false;
        bool is_step_ascii = false;
        
        if (strncmp(buffer.data(), "ISO-10303-21;", 13) == 0) {
            is_step_ascii = true;
            std::cout << "Detected ASCII STEP file\n";
        }
        else if (memmem(buffer.data(), std::min(file_size, 1024L), "solid", 5)) {
            if (memmem(buffer.data(), std::min(file_size, 8192L), "facet normal", 12)) {
                is_stl_ascii = true;
                std::cout << "Detected ASCII STL file\n";
            }
        }
        else {
            is_stl_binary = true;
            std::cout << "Assuming binary STL file\n";
        }
        
        // 3. Rotation matrix components
        float cos_x = std::cos(rot_x * M_PI / 180.0f);
        float sin_x = std::sin(rot_x * M_PI / 180.0f);
        float cos_y = std::cos(rot_y * M_PI / 180.0f);
        float sin_y = std::sin(rot_y * M_PI / 180.0f);
        float cos_z = std::cos(rot_z * M_PI / 180.0f);
        float sin_z = std::sin(rot_z * M_PI / 180.0f);
        
        // 4. Process based on file type
        std::vector<std::array<float, 3>> points;
        int total_count = 0;
        
        if (is_step_ascii) {
            // Extract points from STEP file
            #pragma omp parallel
            {
                std::vector<std::array<float, 3>> local_points;
                local_points.reserve(1000);
                
                #pragma omp for schedule(guided, 1024) nowait
                for (long i = 0; i < file_size - 20; ++i) {
                    if (i + 20 < file_size && 
                        strncmp(&buffer[i], "CARTESIAN_POINT", 15) == 0) {
                        
                        long pos = i;
                        while (pos < file_size && buffer[pos] != '(') pos++;
                        if (pos >= file_size) continue;
                        
                        pos++;
                        while (pos < file_size && buffer[pos] != '(') pos++;
                        if (pos >= file_size) continue;
                        
                        float x, y, z;
                        int count = sscanf(&buffer[pos+1], "%f,%f,%f", &x, &y, &z);
                        if (count == 3) {
                            local_points.push_back({x, y, z});
                        }
                    }
                }
                
                #pragma omp critical
                {
                    points.insert(points.end(), local_points.begin(), local_points.end());
                }
            }
        }
        else if (is_stl_ascii) {
            // Extract vertices from ASCII STL
            #pragma omp parallel
            {
                std::vector<std::array<float, 3>> local_vertices;
                local_vertices.reserve(1000);
                
                #pragma omp for schedule(guided, 1024) nowait
                for (long i = 0; i < file_size - 20; ++i) {
                    if (i + 20 < file_size && 
                        strncmp(&buffer[i], "vertex ", 7) == 0) {
                        
                        float x, y, z;
                        int count = sscanf(&buffer[i+7], "%f %f %f", &x, &y, &z);
                        if (count == 3) {
                            local_vertices.push_back({x, y, z});
                        }
                    }
                }
                
                #pragma omp critical
                {
                    points.insert(points.end(), local_vertices.begin(), local_vertices.end());
                }
            }
        }
        else {  // Binary STL
            unsigned int num_triangles = *reinterpret_cast<unsigned int*>(&buffer[80]);
            const size_t expected_size = 84 + num_triangles * 50;
            
            if (expected_size > static_cast<size_t>(file_size)) {
                throw std::runtime_error("Invalid STL file: triangle count exceeds file size");
            }
            
            // Extract vertices from triangles
            #pragma omp parallel
            {
                std::vector<std::array<float, 3>> local_vertices;
                local_vertices.reserve(num_triangles * 3);
                
                #pragma omp for schedule(guided, 1024) nowait
                for (unsigned int i = 0; i < num_triangles; ++i) {
                    unsigned int offset = 84 + i * 50;
                    if (offset + 50 > static_cast<unsigned int>(file_size)) continue;
                    
                    float vertex[3][3];
                    memcpy(vertex[0], &buffer[offset + 12], 12);
                    memcpy(vertex[1], &buffer[offset + 24], 12);
                    memcpy(vertex[2], &buffer[offset + 36], 12);
                    
                    for (int v = 0; v < 3; ++v) {
                        local_vertices.push_back({vertex[v][0], vertex[v][1], vertex[v][2]});
                    }
                }
                
                #pragma omp critical
                {
                    points.insert(points.end(), local_vertices.begin(), local_vertices.end());
                }
            }
        }
        
        std::cout << "Extracted " << points.size() << " points from 3D file\n";
        
        if (points.empty()) {
            std::cout << "No points extracted - check file format\n";
            return;
        }
        
        // 5. Apply transformations: scale and rotate
        #pragma omp parallel for
        for (size_t i = 0; i < points.size(); ++i) {
            float x = points[i][0] * scale;
            float y = points[i][1] * scale;
            float z = points[i][2] * scale;
            
            // Apply Z rotation
            float x1 = x * cos_z - y * sin_z;
            float y1 = x * sin_z + y * cos_z;
            
            // Apply Y rotation
            float x2 = x1 * cos_y + z * sin_y;
            float z1 = -x1 * sin_y + z * cos_y;
            
            // Apply X rotation
            float y2 = y1 * cos_x - z1 * sin_x;
            float z2 = y1 * sin_x + z1 * cos_x;
            
            points[i][0] = x2;
            points[i][1] = y2;
            points[i][2] = z2;
        }
        
        // 6. Apply surface smoothing (Laplacian smoothing)
        if (smoothing_iterations > 0) {
            std::cout << "Applying Laplacian smoothing (" << smoothing_iterations << " iterations)...\n";
            
            // Build spatial index for neighbor search
            // Using a simple grid-based approach for performance
            const float search_radius = 2.0f * scale;  // Adjust based on model size
            const float grid_cell_size = search_radius;
            
            // Find bounds to create grid
            float min_x = points[0][0], max_x = points[0][0];
            float min_y = points[0][1], max_y = points[0][1];
            float min_z = points[0][2], max_z = points[0][2];
            
            for (const auto& p : points) {
                min_x = std::min(min_x, p[0]); max_x = std::max(max_x, p[0]);
                min_y = std::min(min_y, p[1]); max_y = std::max(max_y, p[1]);
                min_z = std::min(min_z, p[2]); max_z = std::max(max_z, p[2]);
            }
            
            // Create 3D grid for neighbor search
            int grid_x = static_cast<int>((max_x - min_x) / grid_cell_size) + 1;
            int grid_y = static_cast<int>((max_y - min_y) / grid_cell_size) + 1;
            int grid_z = static_cast<int>((max_z - min_z) / grid_cell_size) + 1;
            
            // Use hash map for sparse grid
            std::unordered_map<size_t, std::vector<size_t>> grid;
            
            auto get_grid_key = [&](int ix, int iy, int iz) -> size_t {
                return (static_cast<size_t>(ix) << 20) | (static_cast<size_t>(iy) << 10) | static_cast<size_t>(iz);
            };
            
            auto point_to_grid = [&](const std::array<float, 3>& p) -> std::array<int, 3> {
                return {
                    static_cast<int>((p[0] - min_x) / grid_cell_size),
                    static_cast<int>((p[1] - min_y) / grid_cell_size),
                    static_cast<int>((p[2] - min_z) / grid_cell_size)
                };
            };
            
            // Populate grid
            for (size_t i = 0; i < points.size(); ++i) {
                auto cell = point_to_grid(points[i]);
                grid[get_grid_key(cell[0], cell[1], cell[2])].push_back(i);
            }
            
            // Apply smoothing iterations
            for (int iter = 0; iter < smoothing_iterations; ++iter) {
                std::vector<std::array<float, 3>> new_points = points;
                
                #pragma omp parallel for
                for (size_t i = 0; i < points.size(); ++i) {
                    const auto& p = points[i];
                    auto cell = point_to_grid(p);
                    
                    std::vector<size_t> neighbors;
                    // Search in 3x3x3 neighborhood
                    for (int dx = -1; dx <= 1; ++dx) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dz = -1; dz <= 1; ++dz) {
                                auto key = get_grid_key(cell[0] + dx, cell[1] + dy, cell[2] + dz);
                                auto it = grid.find(key);
                                if (it != grid.end()) {
                                    neighbors.insert(neighbors.end(), it->second.begin(), it->second.end());
                                }
                            }
                        }
                    }
                    
                    // Find neighbors within search radius
                    std::array<float, 3> avg_pos = {0, 0, 0};
                    int neighbor_count = 0;
                    
                    for (size_t j : neighbors) {
                        if (j == i) continue;
                        
                        const auto& p2 = points[j];
                        float dx = p[0] - p2[0];
                        float dy = p[1] - p2[1];
                        float dz = p[2] - p2[2];
                        float dist_sq = dx*dx + dy*dy + dz*dz;
                        
                        if (dist_sq < search_radius * search_radius) {
                            avg_pos[0] += p2[0];
                            avg_pos[1] += p2[1];
                            avg_pos[2] += p2[2];
                            neighbor_count++;
                        }
                    }
                    
                    // Apply smoothing only if we have neighbors
                    if (neighbor_count > 0) {
                        avg_pos[0] /= neighbor_count;
                        avg_pos[1] /= neighbor_count;
                        avg_pos[2] /= neighbor_count;
                        
                        // Laplacian smoothing: new_pos = (1-weight)*old_pos + weight*avg_neighbor_pos
                        float weight = 0.5f;  // Smoothing factor
                        new_points[i][0] = (1.0f - weight) * p[0] + weight * avg_pos[0];
                        new_points[i][1] = (1.0f - weight) * p[1] + weight * avg_pos[1];
                        new_points[i][2] = (1.0f - weight) * p[2] + weight * avg_pos[2];
                    }
                }
                
                points = std::move(new_points);
            }
        }
        
        // 7. Add obstacles with offset
        #pragma omp parallel
        {
            int local_count = 0;
            
            #pragma omp for schedule(guided, 1024) nowait
            for (size_t i = 0; i < points.size(); ++i) {
                float x = points[i][0];
                float y = points[i][1];
                float z = points[i][2];
                
                int sim_x = static_cast<int>(x_offset + x);
                int sim_y = static_cast<int>(y_offset + y);
                int sim_z = static_cast<int>(z_offset + z);
                
                if (sim_x > 0 && sim_x < this->width &&
                    sim_y > 0 && sim_y < this->height &&
                    sim_z > 0 && sim_z < this->depth) {
                    
                    #pragma omp atomic write
                    obs[global_at(sim_x, sim_y, sim_z)] = 1.0f;
                    
                    local_count++;
                }
            }
            
            #pragma omp atomic
            total_count += local_count;
        }
        
        std::cout << "loaded " << total_count << " obstacle points from 3D file\n";
        std::cout << "Rotation: X=" << rot_x << "°, Y=" << rot_y << "°, Z=" << rot_z << "°\n";
        std::cout << "Scale: " << scale << "x\n";
        if (smoothing_iterations > 0) {
            std::cout << "Applied " << smoothing_iterations << " smoothing iterations\n";
        }
    }
};

struct Vec3 { float x, y, z; };

// Helper
std::vector<Vec3> filledSphere(
    int cx, int cy, int cz,
    int radius)
{
    std::vector<Vec3> points;
    int r2 = radius * radius;
    
    // Only iterate over the bounding cube
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            for (int k = -radius; k <= radius; ++k) {
                // Check if inside sphere
                if (i*i + j*j + k*k <= r2) {
                    points.push_back({static_cast<float>(cx + i), 
                                     static_cast<float>(cy + j), 
                                     static_cast<float>(cz + k)});
                }
            }
        }
    }
    return points;
}

/*====================================================================
 *  MAIN
 *====================================================================*/
int main()
{
    /* grid size (without padding) */
    int width  = 128;
    int height = 64;
    int depth  = 64;

    int iter   = 1;
    int speed  = 30;

    Simulation sim(width, height, depth, iter, speed);

    sim.loadObject(32, 16, 16, "/media/raoul/Speed/Data/3D-Printing/Models/Bike/CyclingMount.stl", 1, 0, 0, 0, 2);

    // std::vector<Vec3> sphere = filledSphere(64, 32, 32, 30);
    // for(size_t i = 0; i < sphere.size(); i++) {
    //     sim.addObstacle((int)sphere[i].x, (int)sphere[i].y, (int)sphere[i].z);
    // }

    sim.run();
    return 0;
}