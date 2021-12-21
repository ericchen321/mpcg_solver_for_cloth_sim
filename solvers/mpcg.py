import numpy as np

class MPCGSolver:
    r"""
    Solve system Ax = b with the MPCG method.
    """

    def __init__(self, A_in, b_in, S_in, z_in) -> None:
        r"""
        Constructor for MPCGSolver

        :param A_in: A matrix of shape (3n, 3n)
        :param b_in: vector b of shape (3n, )
        :param S_in: constraint list of length n; each element
            should be a tuple of 0/1/2/3 (not-necessarily unitary)
            vectors indicating prohibited directions
        :param z_in: constrained velocity matrix of shape (n, 3);
            zero vectors for unconstrained particles
        """
        self._A = A_in.copy()
        self._b = np.expand_dims(b_in, axis=1)
        self._z = z_in.copy()
        self._num_particles = self._z.shape[0]
        self._S = self.compute_S(S_in)
        self.compute_M()
        self._epsi = 1e-12

    @property
    def num_particles(self):
        r"""
        Getter for num_particles
        """
        return self._num_particles
    
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
    
    @property
    def P(self):
        r"""
        Get the preconditioner matrix M.
        """
        return self._M

    def compute_M(self):
        r"""
        Compute the preconditioner P from A.
        """
        assert self._A is not None
        self._M = np.diag(np.diag(self._A))

    @property
    def z(self):
        r"""
        Get the constraint matrix z.
        """
        return self._z

    def compute_S(self, S_in):
        r"""
        Compute constraint matrix S for each particle
        from list of constrained direction tuples.

        :param S_in: list of n tuples of prohibited directions

        :return:
            list of n constraint matrices S_i of shape (3, 3)
        """
        S_out = []
        for particle_index in range(self._num_particles):
            # get the constraint vectors and compute S_i
            S_in_i = S_in[particle_index]
            if len(S_in_i) == 0:
                # no constraints
                S_i = np.eye(3)
            elif len(S_in_i) == 1:
                # one constraint
                p = np.expand_dims(S_in_i[0], axis=0)
                S_i = np.eye(3) - np.matmul(np.transpose(p), p)
            elif len(S_in_i) == 2:
                # two constraints
                p = np.expand_dims(S_in_i[0], axis=0)
                q = np.expand_dims(S_in_i[1], axis=0)
                S_i = np.eye(3) - np.matmul(np.transpose(p), p) - np.matmul(np.transpose(q), q)
            else:
                # three constraints
                S_i = np.zeros((3, 3))
            # add S_i to S
            S_out.append(S_i)
        return S_out        

    def filter(self, v):
        r"""
        Filter vector v by kinematic constraints.
        
        :param v: vector of shape (3n, 1)

        :return:
            v filtered by constraints; shape is (3n, 1)
        """
        v_out = np.zeros(v.shape)
        for particle_index in range(self.num_particles):
            # extract v_i
            v_i = v[particle_index*3:(particle_index+1)*3, 0]
            # compute S_i * v_i
            S_i_v_i = np.matmul(self._S[particle_index], v_i)
            # assign S_i*v_i to v_out
            v_out[particle_index*3:(particle_index+1)*3, 0] = S_i_v_i
        return v_out
    
    def solve(self):
        r"""
        Solve A * del_v = b. Return del_v. Algorithm adapted
        from Baraff and Witkin's 98' paper.
        """
        # initialize del_v
        del_v = np.reshape(
            self._z.copy(),
            (self._z.shape[0]*self._z.shape[1], 1))
        
        delta_0 = np.matmul(
            np.transpose(self.filter(self._b)),
            np.matmul(np.linalg.inv(self._M), self.filter(self._b)))
        r = self.filter(self._b - np.matmul(self._A, del_v)) # (3n, 1)
        c = self.filter(np.matmul(np.linalg.inv(self._M), r)) # (3n, 1)
        delta_new = np.matmul(np.transpose(r), c)

        while delta_new > np.power(self._epsi, 2)*delta_0:
            # iterate until relative error in ||r||^2 is small enough
            q = self.filter(np.matmul(self._A, c))
            alpha = delta_new / np.matmul(np.transpose(c), q)
            del_v = del_v + alpha*c
            r = r - alpha*q
            s = np.matmul(np.linalg.inv(self._M), r) # (3n, 1)
            delta_old = delta_new
            delta_new = np.matmul(np.transpose(r), s)
            c = self.filter(s + (delta_new/delta_old)*c)

        return del_v
