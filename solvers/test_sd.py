import numpy as np
from solvers.sd import SDSolver

def test_solve_2d():
    r"""
    Solve a 2D system
    """
    A = np.array([[3, 2], [2, 6]])
    b = np.array([2, -8])
    sd_solver = SDSolver(A, b, 50)
    xs, _ = sd_solver.solve()
    x = xs[-1]
    x_ref = np.matmul(np.linalg.inv(A), np.expand_dims(b, axis=1))
    assert np.linalg.norm(x - x_ref) < 1e-9

if __name__ == "__main__":
    test_solve_2d()