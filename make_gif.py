import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import *

root = "." # path to your data

frames = range(240)
interval=300
repeat_delay = 3000
numb_records = len(frames)

sensor = "radar"

data_list = []

for i in frames:
    data = np.loadtxt(f"{root}/{sensor}/{i:06d}.csv",dtype=np.float32, skiprows=1)
    labels = readLabels(f"{root}/groundtruth/{i:06d}.csv").reshape(-1,7)
    bbs = [make_boundingbox(label) for label in labels]
    bbs = [np.array([bb[0],bb[2],bb[5],bb[6],bb[0]]) for bb in bbs]

    data_list.append((data, bbs))

fig = plt.figure(figsize=(7,7))
ax = plt.axes(xlim=(-25,25),ylim=(0,50))
scatter = ax.scatter(-data_list[0][0][:,0], data_list[0][0][:,1], c = data_list[0][0][:,3], s =10, vmin = -5.3, vmax = 5.3)
fig.tight_layout()

def update(i):
    mark = (data_list[i][0][:,0] > -25) & (data_list[i][0][:,0] < 25) & \
           (data_list[i][0][:,1] >  -50) & (data_list[i][0][:,1] < 50) & \
           (data_list[i][0][:,2] <  1.5)

    ax.cla()
    ax.set_xlim([-25,25])
    ax.set_ylim([0,50])
    ax.title.set_text(f"Frame: {frames[i]}")
    scatter = ax.scatter(data_list[i][0][mark,0], data_list[i][0][mark,1], c = data_list[i][0][mark,2], vmin=-2.3, vmax = 2.3,cmap = "RdBu")
    for bb in data_list[i][1]:
      plt.plot(bb[:,0],bb[:,1])
    fig.tight_layout()
    print(i)
    return scatter,


numb_records = len(frames)
anim = FuncAnimation(fig, update, interval=interval, frames = 240, blit=True, repeat_delay = repeat_delay)
print("Animated")
anim.save('my_gif.gif', writer='imagemagick')
