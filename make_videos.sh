#!/bin/bash

make
./simulation_parallel.out

#SBATCH -J cloth_gif       # Job name
#SBATCH -A m2_jgu-acccomp  # Account name
#SBATCH -p smp             # Partition name
#SBATCH -t 3               # Time in minutes
#SBATCH -n 1               # Number of tasks
#SBATCH --mem=1G           # Memory in per node
#SBATCH -C anyarch         # Allow broadwell and skylake CPUs

# make pngs from binary data
# module purge
# module load vis/matplotlib/3.2.1-foss-2020a-Python-3.8.2
# module load vis/ImageMagick/7.0.10-35-GCCcore-10.2.0
# module load vis/FFmpeg/7.1-GCCcore-11.2.0 

/media/raoul/Speed/Data/Code/Projects/c++/fluid_simulation/.venv/bin/python /media/raoul/Speed/Data/Code/Projects/c++/fluid_simulation/gui.py

ffmpeg -framerate 60 -r 25 -i pngs/density/%d.png -c:v libx264 density.mp4 -y -loglevel warning
ffmpeg -framerate 60 -r 25 -i pngs/velocity_x/%d.png -c:v libx264 velocity_x.mp4 -y -loglevel warning
ffmpeg -framerate 60 -r 25 -i pngs/velocity_y/%d.png -c:v libx264 velocity_y.mp4 -y -loglevel warning

rm -f pngs/density/*
rm -f pngs/velocity_x/*
rm -f pngs/velocity_y/*