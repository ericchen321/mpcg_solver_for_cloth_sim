# Visualize the SD method, from x_0 to somewhere close
# to x
import matplotlib.pyplot as plt
import numpy as np
from solvers.cg import CGSolver
from quad import f_quad

# define A, b (sampled from Shewchuk's work)
A = np.array([[3, 2], [2, 6]])
b = np.array([2, -8])
c = 0

xx = np.linspace(0, 2.5, 100)
yy = np.linspace(-2.5, 0.5, 100)
X, Y = np.meshgrid(xx, yy, sparse=False, indexing='xy')
Z = np.zeros(X.shape)
for col_index in range(xx.shape[0]):
    for row_index in range(yy.shape[0]):
        # compute f and put it in Z
        x_coord = X[row_index, col_index]
        y_coord = Y[row_index, col_index]
        Z[row_index, col_index] = f_quad(A, b, c, np.array([[x_coord], [y_coord]]))

# plot f's contour
plt.contour(X,Y,Z)

# plot x's and r's
cg_solver = CGSolver(A, b, 50)
x_is, r_is = cg_solver.solve()
x_coords = [x_i[0] for x_i in x_is]
y_coords = [x_i[1] for x_i in x_is]
dxs = []
dys = []
for i in range(len(x_coords)-1):
    dx = x_coords[i+1] - x_coords[i]
    dy = y_coords[i+1] - y_coords[i]
    dxs.append(dx)
    dys.append(dy)
plt.scatter(x_coords, y_coords)
for i in range(len(x_coords)-1):
    plt.arrow(
        x_coords[i][0],
        y_coords[i][0],
        dxs[i][0],
        dys[i][0],
        head_width=0.1,
        length_includes_head=True
    )
plt.show()