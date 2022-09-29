import numpy as np
from preprocessor import DataProcessor


class PopeDataProcessor(DataProcessor):
    """
    Inherits from DataProcessor class.  This class is specific to processing turbulence data to predict
    the anisotropy tensor based on the mean strain rate (Sij) and mean rotation rate (Rij) tensors
    """
    @staticmethod
    def calc_Sij_Rij(grad_u, tke, eps, cap=7.):
        """
        Calculates the strain rate and rotation rate tensors.  Normalizes by k and eps:
        Sij = k/eps * 0.5* (grad_u  + grad_u^T)
        Rij = k/eps * 0.5* (grad_u  - grad_u^T)
        :param grad_u: num_points X 3 X 3
        :param tke: turbulent kinetic energy
        :param eps: turbulent dissipation rate epsilon
        :param cap: This is the max magnitude that Sij or Rij components are allowed.  Greater values
                    are capped at this level
        :return: Sij, Rij: num_points X 3 X 3 tensors
        """

        num_points = grad_u.shape[0]
        eps = np.maximum(eps, 1e-8)
        tke_eps = tke / eps
        Sij = np.zeros((num_points, 3, 3))
        Rij = np.zeros((num_points, 3, 3))
        for i in range(num_points):
            Sij[i, :, :] = tke_eps[i] * 0.5 * (grad_u[i, :, :] + np.transpose(grad_u[i, :, :]))
            Rij[i, :, :] = tke_eps[i] * 0.5 * (grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))

        Sij[Sij > cap] = cap
        Sij[Sij < -cap] = -cap
        Rij[Rij > cap] = cap
        Rij[Rij < -cap] = -cap

        # Because we enforced limits on maximum Sij values, we need to re-enforce trace of 0
        for i in range(num_points):
            Sij[i, :, :] = Sij[i, :, :] - 1./3. * np.eye(3)*np.trace(Sij[i, :, :])
        return Sij, Rij

    def calc_scalar_basis(self, Sij, Rij, is_train=False, cap=2.0, is_scale=True):
        """
        Given the non-dimensionalized mean strain rate and mean rotation rate tensors Sij and Rij,
        this returns a set of normalized scalar invariants
        :param Sij: k/eps * 0.5 * (du_i/dx_j + du_j/dx_i)
        :param Rij: k/eps * 0.5 * (du_i/dx_j - du_j/dx_i)
        :param is_train: Determines whether normalization constants should be reset
                        --True if it is training, False if it is test set
        :param cap: Caps the max value of the invariants after first normalization pass
        :return: invariants: The num_points X num_scalar_invariants numpy matrix of scalar invariants
        >>> A = np.zeros((1, 3, 3))
        >>> B = np.zeros((1, 3, 3))
        >>> A[0, :, :] = np.eye(3) * 2.0
        >>> B[0, 1, 0] = 1.0
        >>> B[0, 0, 1] = -1.0
        >>> tdp = PopeDataProcessor()
        >>> tdp.mu = 0
        >>> tdp.std = 0
        >>> scalar_basis = tdp.calc_scalar_basis(A, B, is_scale=False)
        >>> print scalar_basis
        [[ 12.  -2.  24.  -4.  -8.]]
        """
        DataProcessor.calc_scalar_basis(self, Sij, Rij, is_train=is_train)
        num_points = Sij.shape[0]
        num_invariants = 5
        invariants = np.zeros((num_points, num_invariants))
        for i in range(num_points):
            invariants[i, 0] = np.trace(np.dot(Sij[i, :, :], Sij[i, :, :]))
            invariants[i, 1] = np.trace(np.dot(Rij[i, :, :], Rij[i, :, :]))
            invariants[i, 2] = np.trace(np.dot(Sij[i, :, :], np.dot(Sij[i, :, :], Sij[i, :, :])))
            invariants[i, 3] = np.trace(np.dot(Rij[i, :, :], np.dot(Rij[i, :, :], Sij[i, :, :])))
            invariants[i, 4] = np.trace(np.dot(np.dot(Rij[i, :, :], Rij[i, :, :]), np.dot(Sij[i, :, :], Sij[i, :, :])))

        # Renormalize invariants using mean and standard deviation:
        if is_scale:
            if self.mu is None or self.std is None:
                is_train = True
            if is_train:
                self.mu = np.zeros((num_invariants, 2))
                self.std = np.zeros((num_invariants, 2))
                self.mu[:, 0] = np.mean(invariants, axis=0)
                self.std[:, 0] = np.std(invariants, axis=0)

            invariants = (invariants - self.mu[:, 0]) / self.std[:, 0]
            invariants[invariants > cap] = cap  # Cap max magnitude
            invariants[invariants < -cap] = -cap
            invariants = invariants * self.std[:, 0] + self.mu[:, 0]
            if is_train:
                self.mu[:, 1] = np.mean(invariants, axis=0)
                self.std[:, 1] = np.std(invariants, axis=0)

            invariants = (invariants - self.mu[:, 1]) / self.std[:, 1]  # Renormalize a second time after capping
        return invariants

    @staticmethod
    def calc_tensor_basis(Sij, Rij, num_tensor_basis=10, is_scale=True):
        """
        Given Sij and Rij, it calculates the tensor basis
        :param Sij: normalized strain rate tensor
        :param Rij: normalized rotation rate tensor
        :param num_tensor_basis: number of tensor bases to predict
        :return: T_flat: num_points X num_tensor_basis X 9 numpy array of tensor basis.
                        Ordering is 11, 12, 13, 21, 22, ...
        """
        num_points = Sij.shape[0]
        T = np.zeros((num_points, num_tensor_basis, 3, 3))

        # Insert tensor basis functions into a dictionary
        def t1(T, i, sij):
            T[i, 0, :, :] = sij
            return T

        def t2(T, i, sij, rij):
            T[i, 1, :, :] = np.dot(sij, rij) - np.dot(rij, sij)
            return T

        def t3(T, i, sij):
            T[i, 2, :, :] = np.dot(sij, sij) - 1. / 3. * np.eye(3) * np.trace(np.dot(sij, sij))
            return T

        def t4(T, i, rij):
            T[i, 3, :, :] = np.dot(rij, rij) - 1. / 3. * np.eye(3) * np.trace(np.dot(rij, rij))
            return T

        def t5(T, i, sij, rij):
            T[i, 4, :, :] = np.dot(rij, np.dot(sij, sij)) - np.dot(np.dot(sij, sij), rij)
            return T

        def t6(T, i, sij, rij):
            T[i, 5, :, :] = np.dot(rij, np.dot(rij, sij)) + np.dot(sij, np.dot(rij, rij)) \
                            - 2. / 3. * np.eye(3) * np.trace(np.dot(sij, np.dot(rij, rij)))
            return T

        def t7(T, i, sij, rij):
            T[i, 6, :, :] = np.dot(np.dot(rij, sij), np.dot(rij, rij)) - np.dot(np.dot(rij, rij), np.dot(sij, rij))
            return T

        def t8(T, i, sij, rij):
            T[i, 7, :, :] = np.dot(np.dot(sij, rij), np.dot(sij, sij)) - np.dot(np.dot(sij, sij), np.dot(rij, sij))
            return T

        def t9(T, i, sij, rij):
            T[i, 8, :, :] = np.dot(np.dot(rij, rij), np.dot(sij, sij)) + np.dot(np.dot(sij, sij), np.dot(rij, rij)) \
                            - 2. / 3. * np.eye(3) * np.trace(np.dot(np.dot(sij, sij), np.dot(rij, rij)))
            return T

        def t10(T, i, sij, rij):
            T[i, 9, :, :] = np.dot(np.dot(rij, np.dot(sij, sij)), np.dot(rij, rij)) \
                            - np.dot(np.dot(rij, np.dot(rij, sij)), np.dot(sij, rij))
            return T

        tensor_basis_dict = {}
        for j, t in enumerate([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]):
            tensor_basis_dict[j] = t

        for i in range(num_points):
            sij = Sij[i, :, :]
            rij = Rij[i, :, :]
            for j in range(num_tensor_basis):
                T = tensor_basis_dict[j](T, i, sij=sij, rij=rij)
                # Enforce zero trace for anisotropy
                T[i, j, :, :] = T[i, j, :, :] - 1./3.*np.eye(3)*np.trace(T[i, j, :, :])

        # Scale down to promote convergence
        if is_scale:
            scale_factor = [10, 100, 100, 100, 1000, 1000, 10000, 10000, 10000, 10000]
            for j in range(num_tensor_basis):
                T[:, j, :, :] /= scale_factor[j]

        # Flatten:
        T_flat = np.zeros((num_points, num_tensor_basis, 9))
        for i in range(3):
            for j in range(3):
                T_flat[:, :, 3*i+j] = T[:, :, i, j]
        return T_flat

    @staticmethod
    def calc_output(tauij):
        """
        Given Reynolds stress tensor (num_points X 3 X 3), return flattened non-dimensional anisotropy tensor
        :param tauij: Reynolds stress tensor
        :return: anisotropy_flat: (num_points X 9) anisotropy tensor.  bij = (uiuj)/2k - 1./3. * delta_ij
        """
        num_points = tauij.shape[0]
        bij = np.zeros((num_points, 3, 3))

        for i in range(3):
            for j in range(3):
                tke = 0.5 * (tauij[:, 0, 0] + tauij[:, 1, 1] + tauij[:, 2, 2])
                tke = np.maximum(tke, 1e-8)
                bij[:, i, j] = tauij[:, i, j]/(2.0 * tke)
            bij[:, i, i] -= 1./3.
        bij_flat = np.zeros((num_points, 9))
        for i in range(3):
            for j in range(3):
                bij_flat[:, 3*i+j] = bij[:, i, j]
        return bij_flat

    @staticmethod
    def calc_rans_anisotropy(grad_u, tke, eps):
        """
        Calculate the Reynolds stress anisotropy tensor (num_points X 9) that RANS would have predicted
        given a linear eddy viscosity hypothesis: a_ij = -2*nu_t*Sij/(2*k) = - C_mu * k / eps * Sij
        :param grad_u: velocity gradient tensor
        :param tke: turbulent kinetic energy
        :param eps: turbulent dissipation rate
        :return: rans_anisotropy
        """
        sij, _ = PopeDataProcessor.calc_Sij_Rij(grad_u, tke, eps, cap=np.infty)
        c_mu = 0.09

        # Calculate anisotropy tensor (num_points X 3 X 3)
        # Note: Sij is already non-dimensionalized with tke/eps
        rans_bij_matrix = - c_mu * sij

        # Flatten into num_points X 9 array
        num_points = sij.shape[0]
        rans_bij = np.zeros((num_points, 9))
        for i in range(3):
            for j in range(3):
                rans_bij[:, i*3+j] = rans_bij_matrix[:, i, j]
        return rans_bij

    @staticmethod
    def make_realizable(labels):
        """
        This function is specific to turbulence modeling.
        Given the anisotropy tensor, this function forces realizability
        by shifting values within acceptable ranges for Aii > -1/3 and 2|Aij| < Aii + Ajj + 2/3
        Then, if eigenvalues negative, shifts them to zero. Noteworthy that this step can undo
        constraints from first step, so this function should be called iteratively to get convergence
        to a realizable state.
        :param labels: the predicted anisotropy tensor (num_points X 9 array)
        """
        numPoints = labels.shape[0]
        A = np.zeros((3, 3))
        for i in range(numPoints):
            # Scales all on-diags to retain zero trace
            if np.min(labels[i, [0, 4, 8]]) < -1./3.:
                labels[i, [0, 4, 8]] *= -1./(3.*np.min(labels[i, [0, 4, 8]]))
            if 2.*np.abs(labels[i, 1]) > labels[i, 0] + labels[i, 4] + 2./3.:
                labels[i, 1] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
                labels[i, 3] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
            if 2.*np.abs(labels[i, 5]) > labels[i, 4] + labels[i, 8] + 2./3.:
                labels[i, 5] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
                labels[i, 7] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
            if 2.*np.abs(labels[i, 2]) > labels[i, 0] + labels[i, 8] + 2./3.:
                labels[i, 2] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])
                labels[i, 6] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])

            # Enforce positive semidefinite by pushing evalues to non-negative
            A[0, 0] = labels[i, 0]
            A[1, 1] = labels[i, 4]
            A[2, 2] = labels[i, 8]
            A[0, 1] = labels[i, 1]
            A[1, 0] = labels[i, 1]
            A[1, 2] = labels[i, 5]
            A[2, 1] = labels[i, 5]
            A[0, 2] = labels[i, 2]
            A[2, 0] = labels[i, 2]
            evalues, evectors = np.linalg.eig(A)
            if np.max(evalues) < (3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/2.:
                evalues = evalues*(3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/(2.*np.max(evalues))
                A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
                for j in range(3):
                    labels[i, j] = A[j, j]
                labels[i, 1] = A[0, 1]
                labels[i, 5] = A[1, 2]
                labels[i, 2] = A[0, 2]
                labels[i, 3] = A[0, 1]
                labels[i, 7] = A[1, 2]
                labels[i, 6] = A[0, 2]
            if np.max(evalues) > 1./3. - np.sort(evalues)[1]:
                evalues = evalues*(1./3. - np.sort(evalues)[1])/np.max(evalues)
                A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
                for j in range(3):
                    labels[i, j] = A[j, j]
                labels[i, 1] = A[0, 1]
                labels[i, 5] = A[1, 2]
                labels[i, 2] = A[0, 2]
                labels[i, 3] = A[0, 1]
                labels[i, 7] = A[1, 2]
                labels[i, 6] = A[0, 2]

        return labels
