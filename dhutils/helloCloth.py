# %%
import math
import numpy as np
import dhutils as dhu
import pythreejs as THREE

# %%
x0, faces = dhu.standard_rectangle(1.5, 1.5, 10, 10)

# "simulation"
Nt = 3
dt = 0.1
times = []
xt = []

for k in range(0, Nt + 1, 1):
    x = np.copy(x0)
    t = k * dt
    for xj in x:
        xj[2] = math.sin(xj[1]) * math.sin(t)
    xt.append(x)
    times.append(t)

dhu.mesh_animation(times, xt, faces)
