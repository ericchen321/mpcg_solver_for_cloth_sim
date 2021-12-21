import numpy as np

class SDSolver:
    r"""
    Solve system Ax = b with the Method of Steepest Descent (SD).
    """

    def __init__(self, A_in, b_in, i_max_in) -> None:
        r"""
        Constructor for SDSolver

        :param A_in: A matrix of shape (n, n)
        :param b_in: vector b of shape (n, )
        :param i_max_in: max number of iterations when solving
        """
        self._A = A_in.copy()
        self._b = np.expand_dims(b_in, axis=1)
        self._i_max = i_max_in
        self._x_0 = np.zeros(self._b.shape)
        self._epsilon = 1e-12

    def solve(self):
        r"""
        Solve A * x = b. Method adapted from Shewchuk's introductory
        text on CG from 94'.
        
        Return:
        - a list of x_i's ((n, 1) array each) from the initial guess x_0
        to the approximated solution x_final;
        - a list of r_i's
        """
        i = 0
        x_i = self._x_0.copy()
        x_is = [x_i]
        r = self._b - np.matmul(self._A, x_i)
        r_is = [r]
        delta = np.matmul(np.transpose(r), r)
        delta_0 = delta
        while i<self._i_max and delta>np.power(self._epsilon, 2)*delta_0:
            q = np.matmul(self._A, r)
            alpha = delta / (np.matmul(np.transpose(r), q))
            # update x_i
            x_i = x_i + alpha*r
            # update r
            r = self._b - np.matmul(self._A, x_i)
            delta = np.matmul(np.transpose(r), r)
            i = i + 1
            x_is.append(x_i)
            r_is.append(r)
        return x_is, r_is