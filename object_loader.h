// object_loader.h
#ifndef OBJECT_LOADER_H
#define OBJECT_LOADER_H

class Simulation;

void loadSTLIntoObstacles(
    const char* stlFile,
    Simulation& sim,
    float scale = 0.8f,
    float rot_x = 0.0f,
    float rot_y = 0.0f,
    float rot_z = 0.0f,
    float translate_x = 0.0f,
    float translate_y = 0.0f,
    float translate_z = 0.0f
);

#endif