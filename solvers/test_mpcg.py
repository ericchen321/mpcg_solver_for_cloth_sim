import numpy as np
from solvers.mpcg import MPCGSolver

def test_filter_one_particle_unconstrained():
    r"""
    case: 1 particle unconstrained
    """
    A = np.array([[5.0, -1, 0], [-1, 10, 0], [0, 0, 1]]) # is SPD
    b = np.transpose(np.array([2.0, -8, 3]))
    S = [()]
    z = np.zeros((1, 3))
    mpcg_solver = MPCGSolver(A, b, S, z)
    v = np.expand_dims(np.array([1.0, 2, 3]), axis=1)
    v_filtered = mpcg_solver.filter(v)
    assert np.linalg.norm(v_filtered - v) < 1e-9

def test_filter_one_particle_one_constraint():
    r"""
    case: 1 particle constrained along x
    """
    A = np.array([[5.0, -1, 0], [-1, 10, 0], [0, 0, 1]]) # is SPD
    b = np.transpose(np.array([2.0, -8, 3]))
    S = [(np.array([1, 0, 0]), )]
    z = np.array([[2.9, 0.0, 0.0]]) # does not matter
    mpcg_solver = MPCGSolver(A, b, S, z)
    v = np.expand_dims(np.array([1.0, 2, 3]), axis=1)
    v_filtered = mpcg_solver.filter(v)
    v_filtered_ref = np.expand_dims(np.array([0.0, 2, 3]), axis=1)
    assert np.linalg.norm(v_filtered - v_filtered_ref) < 1e-9

def test_filter_one_particle_two_constraints():
    r"""
    case: 1 particle constrained along x, y
    """
    A = np.array([[5.0, -1, 0], [-1, 10, 0], [0, 0, 1]]) # is SPD
    b = np.transpose(np.array([2.0, -8, 3]))
    S = [(np.array([1, 0, 0]), np.array([0, 1, 0]))]
    z = np.array([[2.9, 0.0, 0.0]]) # does not matter
    mpcg_solver = MPCGSolver(A, b, S, z)
    v = np.expand_dims(np.array([1.0, 2, 3]), axis=1)
    v_filtered = mpcg_solver.filter(v)
    v_filtered_ref = np.expand_dims(np.array([0.0, 0.0, 3]), axis=1)
    assert np.linalg.norm(v_filtered - v_filtered_ref) < 1e-9

def test_filter_one_particle_three_constraints():
    r"""
    case: 1 particle constrained along x, y, z
    """
    A = np.array([[5.0, -1, 0], [-1, 10, 0], [0, 0, 1]]) # is SPD
    b = np.transpose(np.array([2.0, -8, 3]))
    S = [(np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))]
    z = np.array([[2.9, 0.0, 0.0]]) # does not matter
    mpcg_solver = MPCGSolver(A, b, S, z)
    v = np.expand_dims(np.array([1.0, 2, 3]), axis=1)
    v_filtered = mpcg_solver.filter(v)
    v_filtered_ref = np.expand_dims(np.array([0.0, 0.0, 0.0]), axis=1)
    assert np.linalg.norm(v_filtered - v_filtered_ref) < 1e-9

def test_solve_one_particle_unconstrained():
    r"""
    case: 1 particle unconstrained
    """
    A = np.array([[5.0, -1, 0], [-1, 10, 0], [0, 0, 1]]) # is SPD
    b = np.array([2.0, -8, 3])
    S = [()]
    z = np.array([[2.9, 3.1, 7.5]])
    mpcg_solver = MPCGSolver(A, b, S, z)
    x = mpcg_solver.solve()
    x_ref = np.matmul(np.linalg.inv(A), np.expand_dims(b, axis=1))
    assert np.linalg.norm(x - x_ref) < 1e-9

def test_solve_one_particle_one_constraint():
    r"""
    case: 1 particle constrained along x
    """
    A = np.array([[5.0, -1, 0], [-1, 10, 0], [0, 0, 1]]) # is SPD
    b = np.array([2.0, -8, 3])
    S = [(np.array([1, 0, 0]), )]
    z = np.array([[2.9, 3.1, 7.5]])
    mpcg_solver = MPCGSolver(A, b, S, z)
    x = mpcg_solver.solve()
    assert np.linalg.norm(x[0, 0] - z[0, 0]) < 1e-9

def test_solve_one_particle_two_constraints():
    r"""
    case: 1 particle constrained along x and y
    """
    A = np.array([[5.0, -1, 0], [-1, 10, 0], [0, 0, 1]]) # is SPD
    b = np.array([2.0, -8, 3])
    S = [(np.array([1, 0, 0]), np.array([0, 1, 0]))]
    z = np.array([[2.9, 3.1, 7.5]])
    mpcg_solver = MPCGSolver(A, b, S, z)
    x = mpcg_solver.solve()
    assert np.linalg.norm(x[0:2, 0] - np.transpose(z)[0:2, 0]) < 1e-9

if __name__ == "__main__":
    test_filter_one_particle_unconstrained()
    test_filter_one_particle_one_constraint()
    test_filter_one_particle_two_constraints()
    test_filter_one_particle_three_constraints()
    test_solve_one_particle_unconstrained()
    test_solve_one_particle_one_constraint()
    test_solve_one_particle_two_constraints()
