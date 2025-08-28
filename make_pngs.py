import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import os, sys

width = 512 + 2
height = 256 + 2

fwidth = 6
fheight = 3

def progress_bar(current, total, bar_length=50):
    """Display a simple progress bar in the console"""
    percent = float(current) * 100 / total
    arrow = '=' * int(percent / 100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f'\r[{arrow}{spaces}] {percent:.2f}% ({current}/{total})')
    sys.stdout.flush()

# Create a custom colormap: white -> green -> blue -> red
density_cmap = LinearSegmentedColormap.from_list(
    'density_cmap', 
    ['white', 'lightgreen', 'green', 'deepskyblue', 'blue', 'darkred', 'red']
)

plt.rcParams['figure.figsize'] = (fwidth, fheight)

# plot density field
with open("data/data.bin", "rb") as f:
    data = np.fromfile(f, dtype=np.float32)

#plot velocity 
with open("data/v_x.bin", "rb") as f:
    v_x = np.fromfile(f, dtype=np.float32)
with open("data/v_y.bin", "rb") as f:
    v_y = np.fromfile(f, dtype=np.float32)    

with open("data/obs.bin", "rb") as f:
    obs = np.fromfile(f, dtype=np.float32)    

v_x = v_x.reshape(-1,height,width)
v_y = v_y.reshape(-1,height,width)
data = data.reshape(-1,height,width)
obs = obs.reshape(-1, height, width)

x,y = np.meshgrid(np.arange(height), np.arange(width))

z = [data[i,:,:] for i in range(data.shape[0])]
v_x_all = [v_x[i,:,:] for i in range(v_x.shape[0])]
v_y_all = [v_y[i,:,:] for i in range(v_y.shape[0])]

paths = ["density", "velocity_x", "velocity_y"]
current_frame = 0

print("creating images")
for path in paths:
    os.makedirs(f"pngs/{path}",exist_ok=True)

    # print(f"\n{path}")

    for i in range(data.shape[0]):
        plt.tight_layout()
        plt.axis(False)
        if 'velocity_x' in path:
            plt.imshow(v_x[i], vmin=-10, vmax=10, cmap='Greys')
        elif 'velocity_y' in path:
            plt.imshow(v_y[i], vmin=-1, vmax=1, cmap='Greys')
        else:
            # Use custom density colormap with appropriate scaling
            plt.imshow(data[i], vmin=0, vmax=0.01, cmap=density_cmap)
        
        plt.imshow(obs[i], cmap='binary', alpha=0.1)
        plt.savefig(f"pngs/{path}/{i}.png")
        plt.clf()
        
        progress_bar(current_frame, (data.shape[0] * 3) - 1)
        current_frame += 1
        # if (i + 1) % 100 == 0 and i > 0:
        #     print(f"created {i + 1} images")

print("")