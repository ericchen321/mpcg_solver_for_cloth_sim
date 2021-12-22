import numpy as np

class CGSolver:
    r"""
    Solve system Ax = b with the CG method.
    """

    def __init__(self, A_in, b_in, i_max_in) -> None:
        r"""
        Constructor for PCGSolver

        :param A_in: A matrix of shape (n, n)
        :param b_in: vector b of shape (n, )
        :param i_max_in: max number of iterations when solving
        """
        self._A = A_in.copy()
        self._b = np.expand_dims(b_in, axis=1)
        self._i_max = i_max_in
        self._x_0 = np.zeros(self._b.shape)
        self._epsilon = 1e-12
    
    @property
    def A(self):
        r"""
        Getter for A
        """
        return self._A

    @property
    def b(self):
        r"""
        Getter for b
        """
        return self._b
    
    def solve(self):
        r"""
        Solve A * x = b. Return x and direction of descent d at each
        step. Algorithm adapted from Shewchuk, 94'.

        Return:
        - a list of x_i's ((n, 1) array each) from the initial guess x_0
        to the approximated solution x_final;
        - a list of d_i's
        """
        i = 0
        x_i = self._x_0.copy()
        x_is = [x_i]
        r = self._b - np.matmul(self._A, x_i)
        d = r
        d_is = [d]
        delta_new = np.matmul(np.transpose(r), r)
        delta_0 = delta_new

        while i<self._i_max and delta_new>np.power(self._epsilon, 2)*delta_0:
            q = np.matmul(self._A, d)
            alpha = delta_new / np.matmul(np.transpose(d), q)
            # update x_i
            x_i = x_i + alpha * d
            # update r
            r = self._b - np.matmul(self._A, x_i)
            delta_old = delta_new
            delta_new = np.matmul(np.transpose(r), r)
            beta = delta_new/delta_old
            # update d
            d = r + beta*d
            i = i + 1
            x_is.append(x_i)
            d_is.append(d)
        return x_is, d_is