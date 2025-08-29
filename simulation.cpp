// simulation.cpp
#include "simulation.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <cstring>
#include <cmath>

/*====================================================================
 *  Simulation Class Implementation
 *====================================================================*/

/*--------------------------------------------------------------
 *  Constructor – now takes depth as an extra argument
 *--------------------------------------------------------------*/
Simulation::Simulation(int w, int h, int d, int iter,
                       int speed,
                       float dt,
                       float diff,
                       float visc,
                       int acc)
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
    size = (width + 2) * (height + 2) * (depth + 2);

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
void Simulation::run()
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

        // Copy density to buffer
        buffer = dens;
        step();

        if ((i + 1) % 100 == 0 && i > 0) {
            std::cout << "step " << i + 1 << "\n";
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
void Simulation::step()
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

        /* diffuse */
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
            f_dens_data.write(reinterpret_cast<char*>(dens.data()), sizeof(float) * size);
            f_dens_obs .write(reinterpret_cast<char*>(obs .data()), sizeof(float) * size);
            f_vel_x    .write(reinterpret_cast<char*>(v_x .data()), sizeof(float) * size);
            f_vel_y    .write(reinterpret_cast<char*>(v_y .data()), sizeof(float) * size);
            f_vel_z    .write(reinterpret_cast<char*>(v_z .data()), sizeof(float) * size);
        }
    }
}

/*--------------------------------------------------------------------
 *  Helper: add a single obstacle cell (3-D overload)
 *--------------------------------------------------------------------*/
void Simulation::addObstacle(int x, int y, int z)
{
    obs[global_at(x, y, z)] = 1.0f;
}

/*--------------------------------------------------------------------
 *  Helper: add density to a cell (3-D overload)
 *--------------------------------------------------------------------*/
void Simulation::addDensity(int x, int y, int z, float amount)
{
    dens[global_at(x, y, z)] += amount;
}

/*--------------------------------------------------------------------
 *  Helper: set velocity at a cell (3-D overload)
 *--------------------------------------------------------------------*/
void Simulation::setVelocity(int x, int y, int z,
                             float amount_x, float amount_y, float amount_z)
{
    int pos = global_at(x, y, z);
    v_x[pos] = amount_x;
    v_y[pos] = amount_y;
    v_z[pos] = amount_z;
}

/*--------------------------------------------------------------------
 *  Boundary handling – works for x, y, and z faces
 *--------------------------------------------------------------------*/
void Simulation::setBounds(int b, std::vector<float>& field)
{
    /* ----- x-faces (left/right) ----- */
    #pragma omp for
    for (int y = 1; y <= height; ++y)
        for (int z = 1; z <= depth; ++z) {
            field[global_at(0, y, z)] = (b == 1) ? -field[global_at(1, y, z)]
                                                : field[global_at(1, y, z)];
            field[global_at(width+1, y, z)] = field[global_at(width, y, z)]; // outflow
        }

    /* ----- y-faces (bottom/top) ----- */
    #pragma omp for
    for (int x = 1; x <= width; ++x)
        for (int z = 1; z <= depth; ++z) {
            field[global_at(x, 0, z)] = (b == 2) ? -field[global_at(x, 1, z)]
                                                : field[global_at(x, 1, z)];
            field[global_at(x, height+1, z)] = (b == 2) ? -field[global_at(x, height, z)]
                                                      : field[global_at(x, height, z)];
        }

    /* ----- z-faces (front/back) ----- */
    #pragma omp for
    for (int x = 1; x <= width; ++x)
        for (int y = 1; y <= height; ++y) {
            field[global_at(x, y, 0)] = (b == 3)
                                      ? -field[global_at(x, y, 1)]
                                      : field[global_at(x, y, 1)];

            field[global_at(x, y, depth+1)] = (b == 3)
                                            ? -field[global_at(x, y, depth)]
                                            : field[global_at(x, y, depth)];
        }

    /* ----- obstacle handling – zero velocity/density inside solids ----- */
    #pragma omp for collapse(3)
    for (int x = 1; x <= width; ++x)
        for (int y = 1; y <= height; ++y)
            for (int z = 1; z <= depth; ++z)
                if (obs[global_at(x, y, z)] == 1.0f)
                    field[global_at(x, y, z)] = 0.0f;

    /* ----- CRITICAL: No-slip condition on fluid cells adjacent to obstacles ----- */
    #pragma omp for collapse(3)
    for (int i = 1; i <= width; ++i) {
        for (int j = 1; j <= height; ++j) {
            for (int k = 1; k <= depth; ++k) {
                if (obs[global_at(i, j, k)] == 1.0f) continue;

                bool isAdjacentToObstacle =
                    (i+1 <= width && obs[global_at(i+1, j, k)] == 1.0f) ||
                    (i-1 >= 1 && obs[global_at(i-1, j, k)] == 1.0f) ||
                    (j+1 <= height && obs[global_at(i, j+1, k)] == 1.0f) ||
                    (j-1 >= 1 && obs[global_at(i, j-1, k)] == 1.0f) ||
                    (k+1 <= depth && obs[global_at(i, j, k+1)] == 1.0f) ||
                    (k-1 >= 1 && obs[global_at(i, j, k-1)] == 1.0f);

                if (isAdjacentToObstacle && (b == 1 || b == 2 || b == 3)) {
                    field[global_at(i, j, k)] = 0.0f;
                }
            }
        }
    }
}

/*--------------------------------------------------------------------
 *  Linear solver – Gauss–Seidel (3D, six neighbors)
 *--------------------------------------------------------------------*/
void Simulation::linearSolver(int b,
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
                          field[global_at(i,   j,   l+1)] + field[global_at(i,   j,   l-1)]))
                        * cRecip;
                }
        setBounds(b, field);
    }
}

/*--------------------------------------------------------------------
 *  Diffusion – wrapper around linear solver
 *--------------------------------------------------------------------*/
void Simulation::diffuse(int b,
                         std::vector<float>& field,
                         std::vector<float>& prev_field)
{
    float a = dt * diff * width * height * depth;
    linearSolver(b, field, prev_field, a, 1.0f + 6.0f * a);
}

/*--------------------------------------------------------------------
 *  Projection – make velocity field divergence-free
 *--------------------------------------------------------------------*/
void Simulation::project(std::vector<float>& v_x,
                         std::vector<float>& v_y,
                         std::vector<float>& v_z,
                         std::vector<float>& p,
                         std::vector<float>& div)
{
    float h = 1.0f / std::cbrt(static_cast<float>(width * height * depth));

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
                if (i+1 <= width && obs[global_at(i+1, j, k)] == 0.0f) div_val += v_x[global_at(i+1, j, k)];
                if (i-1 >= 1   && obs[global_at(i-1, j, k)] == 0.0f) div_val -= v_x[global_at(i-1, j, k)];
                if (j+1 <= height && obs[global_at(i, j+1, k)] == 0.0f) div_val += v_y[global_at(i, j+1, k)];
                if (j-1 >= 1    && obs[global_at(i, j-1, k)] == 0.0f) div_val -= v_y[global_at(i, j-1, k)];
                if (k+1 <= depth && obs[global_at(i, j, k+1)] == 0.0f) div_val += v_z[global_at(i, j, k+1)];
                if (k-1 >= 1    && obs[global_at(i, j, k-1)] == 0.0f) div_val -= v_z[global_at(i, j, k-1)];

                div[global_at(i, j, k)] = -0.5f * h * div_val;
                p[global_at(i, j, k)]   = 0.0f;
            }

    setBounds(0, div);
    setBounds(0, p);
    linearSolver(0, p, div, 1.0f, 6.0f);

    #pragma omp for collapse(3)
    for (int i = 1; i <= width; ++i)
        for (int j = 1; j <= height; ++j)
            for (int k = 1; k <= depth; ++k) {
                if (obs[global_at(i, j, k)] == 1.0f) continue;

                float grad_x = 0.0f;
                if (i+1 <= width && obs[global_at(i+1, j, k)] == 0.0f && i-1 >= 1 && obs[global_at(i-1, j, k)] == 0.0f) {
                    grad_x = (p[global_at(i+1, j, k)] - p[global_at(i-1, j, k)]) / (2.0f * h);
                } else if (i+1 <= width && obs[global_at(i+1, j, k)] == 0.0f) {
                    grad_x = (p[global_at(i+1, j, k)] - p[global_at(i, j, k)]) / h;
                } else if (i-1 >= 1 && obs[global_at(i-1, j, k)] == 0.0f) {
                    grad_x = (p[global_at(i, j, k)] - p[global_at(i-1, j, k)]) / h;
                }
                v_x[global_at(i, j, k)] -= grad_x;

                float grad_y = 0.0f;
                if (j+1 <= height && obs[global_at(i, j+1, k)] == 0.0f && j-1 >= 1 && obs[global_at(i, j-1, k)] == 0.0f) {
                    grad_y = (p[global_at(i, j+1, k)] - p[global_at(i, j-1, k)]) / (2.0f * h);
                } else if (j+1 <= height && obs[global_at(i, j+1, k)] == 0.0f) {
                    grad_y = (p[global_at(i, j+1, k)] - p[global_at(i, j, k)]) / h;
                } else if (j-1 >= 1 && obs[global_at(i, j-1, k)] == 0.0f) {
                    grad_y = (p[global_at(i, j, k)] - p[global_at(i, j-1, k)]) / h;
                }
                v_y[global_at(i, j, k)] -= grad_y;

                float grad_z = 0.0f;
                if (k+1 <= depth && obs[global_at(i, j, k+1)] == 0.0f && k-1 >= 1 && obs[global_at(i, j, k-1)] == 0.0f) {
                    grad_z = (p[global_at(i, j, k+1)] - p[global_at(i, j, k-1)]) / (2.0f * h);
                } else if (k+1 <= depth && obs[global_at(i, j, k+1)] == 0.0f) {
                    grad_z = (p[global_at(i, j, k+1)] - p[global_at(i, j, k)]) / h;
                } else if (k-1 >= 1 && obs[global_at(i, j, k-1)] == 0.0f) {
                    grad_z = (p[global_at(i, j, k)] - p[global_at(i, j, k-1)]) / h;
                }
                v_z[global_at(i, j, k)] -= grad_z;
            }

    setBounds(1, v_x);
    setBounds(2, v_y);
    setBounds(3, v_z);
}

/*--------------------------------------------------------------------
 *  Semi-Lagrangian advection – 3D
 *--------------------------------------------------------------------*/
void Simulation::advect(int b,
                        std::vector<float>& field,
                        std::vector<float>& prev_field)
{
    #pragma omp for collapse(3)
    for (int i = 1; i <= width; ++i)
        for (int j = 1; j <= height; ++j)
            for (int k = 1; k <= depth; ++k) {
                if (obs[global_at(i, j, k)] == 1.0f) {
                    field[global_at(i, j, k)] = 0.0f;
                    continue;
                }

                float vx = (b == 1) ? prev_field[global_at(i,j,k)] : v_x[global_at(i,j,k)];
                float vy = (b == 2) ? prev_field[global_at(i,j,k)] : v_y[global_at(i,j,k)];
                float vz = (b == 3) ? prev_field[global_at(i,j,k)] : v_z[global_at(i,j,k)];

                float x_back = i - dt * width  * vx;
                float y_back = j - dt * height * vy;
                float z_back = k - dt * depth  * vz;

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

/*====================================================================
 *  MAIN
 *====================================================================*/
int main()
{
    int width  = 128;
    int height = 64;
    int depth  = 64;
    int iter   = 1;
    int speed  = 30;

    Simulation sim(width, height, depth, iter, speed);

    // Load obstacle from STL
    loadSTLIntoObstacles("/media/raoul/Speed/Data/3D-Printing/Models/Cars/F1Car-basic.stl", sim, 
                     2.0f,     // scale
                     90.0f,     // rot_x
                     0.0f,    // rot_y
                     0.0f,     // rot_z
                     0.0f, 0.0f, 0.0f); // translate

    sim.run();

    return 0;
}