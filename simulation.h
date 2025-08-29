// simulation.h
#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <fstream>

/* 3-D indexing macro */
#define global_at(x, y, z) ((x) + (y)*(width+2) + (z)*(width+2)*(height+2))

class Simulation {
private:
    int size;  // total number of cells (incl. padding)

    /* velocity fields – current and previous time level */
    std::vector<float> v_x, v_y, v_z;
    std::vector<float> v_x_prev, v_y_prev, v_z_prev;

    /* scalar field (density) and its temporary buffer */
    std::vector<float> dens, buffer;

    /* obstacle mask (1 = solid, 0 = fluid) */
    std::vector<float> obs;

    /* pressure-projection helpers */
    std::vector<float> pressure;   // scalar pressure field (p)
    std::vector<float> divergence; // scalar divergence field (div)

    /* output streams */
    std::ofstream f_dens_data, f_dens_obs, f_vel_x, f_vel_y, f_vel_z;

    /* internal methods */
    void setBounds(int b, std::vector<float>& field);
    void linearSolver(int b, std::vector<float>& field, std::vector<float>& prev_field,
                      float diff_adj, float c);
    void diffuse(int b, std::vector<float>& field, std::vector<float>& prev_field);
    void project(std::vector<float>& v_x, std::vector<float>& v_y,
                 std::vector<float>& v_z, std::vector<float>& p,
                 std::vector<float>& div);
    void advect(int b, std::vector<float>& field, std::vector<float>& prev_field);

public:
    /* grid dimensions (without padding) */
    int width, height, depth;

    /* inlet speed (x-direction) */
    int speed;

    /* simulation parameters */
    float dt;     // time step
    float diff;   // scalar diffusion coefficient
    float visc;   // kinematic viscosity
    int   acc;    // Gauss–Seidel iterations
    int   iter;   // total number of steps

    /*--------------------------------------------------------------
     *  Constructor – takes grid size and simulation parameters
     *--------------------------------------------------------------*/
    Simulation(int w, int h, int d, int iter,
               int speed = 30,
               float dt = 0.05f,
               float diff = 2.0e-5f,
               float visc = 1.5e-5f,
               int acc = 15);

    /*--------------------------------------------------------------
     *  Run the simulation
     *--------------------------------------------------------------*/
    void run();

    /*--------------------------------------------------------------
     *  One simulation step (diffuse → project → advect → project)
     *--------------------------------------------------------------*/
    void step();

    /*--------------------------------------------------------------------
     *  Helper: add a single obstacle cell
     *--------------------------------------------------------------------*/
    void addObstacle(int x, int y, int z);

    /*--------------------------------------------------------------------
     *  Helper: add density to a cell
     *--------------------------------------------------------------------*/
    void addDensity(int x, int y, int z, float amount);

    /*--------------------------------------------------------------------
     *  Helper: set velocity at a cell
     *--------------------------------------------------------------------*/
    void setVelocity(int x, int y, int z,
                     float amount_x, float amount_y, float amount_z);
};

/* Load STL and populate obstacles (declared here, defined in object_loader.cpp) */
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
);

#endif // SIMULATION_H