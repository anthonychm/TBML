import numpy as np
from numpy import trace as tr
from numpy import dot as dot
from numpy.linalg import multi_dot as mdot
import random


class PopeDataProcessor:
    def __init__(self):
        self.mu = None
        self.std = None

    @staticmethod
    def calc_Sij_Rij(grad_u, k, eps, cap_val=7., cap=False, normalize=False, nondim=True):
        # ✓
        """
        Calculates the non-dimensional mean strain rate and rotation rate tensors.
        :param grad_u: num_points X 3 X 3
        :param k: turbulent kinetic energy
        :param eps: turbulent kinetic energy dissipation rate
        :param cap_val: This is the max magnitude that Sij and Rij components are allowed
        :param cap: Boolean for capping the max magnitude of the Sij and Rij components
        :param normalize: Boolean for normalizing Sij and Rij
        :param nondim: Boolean for non-dimensionalising Sij and Rij
        :return: Sij, Rij in the form of num_points X 3 X 3 tensors
        """

        assert normalize != nondim, \
            "Both normalize and nondimensionalize have been set to true or false"
        num_points = grad_u.shape[0]
        eps = np.maximum(eps, 1e-8)
        k_eps = k / eps
        Sij = np.full((num_points, 3, 3), np.nan)
        Rij = np.full((num_points, 3, 3), np.nan)

        # Calculate non-dimensional mean strain and rotation rate tensors ✓
        # Sij = (k/eps) * 0.5 * (grad_u  + grad_u^T)
        # Rij = (k/eps) * 0.5 * (grad_u  - grad_u^T)
        if nondim is True:
            for i in range(num_points):
                Sij[i, :, :] = k_eps[i] * 0.5 * (grad_u[i, :, :] +
                                                 np.transpose(grad_u[i, :, :]))
                Rij[i, :, :] = k_eps[i] * 0.5 * (grad_u[i, :, :] -
                                                 np.transpose(grad_u[i, :, :]))

            if cap is True:
                Sij[Sij > cap_val] = cap_val
                Sij[Sij < -cap_val] = -cap_val
                Rij[Rij > cap_val] = cap_val
                Rij[Rij < -cap_val] = -cap_val

                # Because we enforced limits on maximum Sij values, we need to re-enforce
                # trace of 0
                for i in range(num_points):
                    Sij[i, :, :] = Sij[i, :, :] - 1./3. * np.eye(3)*np.trace(Sij[i, :, :])

        # Calculate normalized mean strain and rotation rate tensors
        # Sij = Sij/(|Sij|+|eps/k|), Rij = Rij/(2*|Rij|)
        if normalize is True:
            for i in range(num_points):
                sij = 0.5 * (grad_u[i, :, :] + np.transpose(grad_u[i, :, :]))
                rij = 0.5 * (grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))
                Sij[i, :, :] = sij/(np.linalg.norm(sij) + (1/k_eps[i]))
                Rij[i, :, :] = rij/(2 * np.linalg.norm(rij))

        return Sij, Rij

    def calc_Ap(self, grad_p, density, u, grad_u, normalize=True):  # ✓

        # Calculate antisymmetric tensor Ap associated with grad_p ✓
        if normalize is True:
            # Normalization from Wang et al. (2018)
            # grad_p_hat = grad_p/(|grad_p|+|rho|*|U*gradU|||)

            beta = np.full((grad_p.shape[0], 1), np.nan)
            grad_p_norm = np.full((grad_p.shape[0], 1), np.nan)
            grad_p_hat = np.full((grad_p.shape[0], 3), np.nan)

            for i in range(grad_p.shape[0]):
                beta[i] = density * np.linalg.norm(np.matmul(u[i, :], grad_u[i, :, :]))
                grad_p_norm[i] = np.linalg.norm(grad_p[i, :])
                grad_p_hat[i, :] = grad_p[i, :]/(grad_p_norm[i] + beta[i]) #2.33e-9

            # Check that all values of grad_p_hat fall between [-1, 1] ✓
            assert np.any(grad_p_hat < -1) == False, \
                "grad_p_hat not constrained between [-1, 1]"
            assert np.any(grad_p_hat > 1) == False, \
                "grad_p_hat not constrained between [-1, 1]"
        else:
            grad_p_hat = grad_p

        # Calculate Ap ✓
        Ap = self.calc_anti(grad_p_hat) #0.114
        return Ap

    def calc_Ak(self, grad_k, eps, k, normalize=True):  # ✓

        # Calculate antisymmetric tensor Ak associated with grad_k ✓
        if normalize is True:
            # Normalization from Wang et al. (2018)
            # grad_k_hat = grad_k/(|grad_k|+|eps/sqrt(k)|)

            beta = eps/np.sqrt(k)
            grad_k_norm = np.full((grad_k.shape[0], 1), np.nan)
            grad_k_hat = np.full((grad_k.shape[0], 3), np.nan)

            for i in range(grad_k.shape[0]):
                grad_k_norm[i] = np.linalg.norm(grad_k[i, :])
                grad_k_hat[i, :] = grad_k[i, :]/(grad_k_norm[i] + beta[i]) #0.738

            # Check that all values of grad_k_hat fall between [-1, 1] ✓
            assert np.any(grad_k_hat < -1) == False, \
                "grad_k_hat not constrained between [-1, 1]"
            assert np.any(grad_k_hat > 1) == False, \
                "grad_k_hat not constrained between [-1, 1]"
        else:
            grad_k_hat = grad_k

        # Calculate Ak ✓
        Ak = self.calc_anti(grad_k_hat)
        return Ak

    @staticmethod
    def calc_anti(hat_vector):  # ✓
        # Recast non-dimensional gradient vector into an antisymmetric tensor ✓
        anti = np.full((hat_vector.shape[0], 3, 3), np.nan)
        for i in range(hat_vector.shape[0]):
            for j in range(3):
                anti[i, j, j] = 0
            anti[i, 0, 1] = -hat_vector[i, 2]
            anti[i, 0, 2] = hat_vector[i, 1]
            anti[i, 1, 0] = hat_vector[i, 2]
            anti[i, 1, 2] = -hat_vector[i, 0]
            anti[i, 2, 0] = -hat_vector[i, 1]
            anti[i, 2, 1] = hat_vector[i, 0]

        # Check antisymmetry ✓
        rand_idx = random.randint(0, hat_vector.shape[0])
        for i in range(3):
            for j in range(3):
                if i == j:
                    pass
                else:
                    assert anti[rand_idx, i, j] == -anti[rand_idx, j, i]
        return anti

    def calc_scalar_basis(self, Sij, Rij, Ap=None, Ak=None, pressure=True, tke=True,
                          is_train=True, standardize=True, cap=2.0, load=False,
                          two_invars=False):  # ✓

        """
        Given Sij, Rij, Ap* and Ak*, this function returns a set of normalized scalar
        invariants.
        * Optional
        :param Sij: k/eps * 0.5 * (du_i/dx_j + du_j/dx_i)
        :param Rij: k/eps * 0.5 * (du_i/dx_j - du_j/dx_i)
        :param Ap: -I x grad_p/(|grad_p|+|rho|*|U*gradU|||)
        :param Ak: -I x grad_k/(|grad_k|+|eps/sqrt(k)|)
        :param pressure: Boolean for including pressure gradient invariants
        :param tke: Boolean for including tke gradient invariants
        :param is_train: Determines whether normalization constants should be reset
                        --True if it is training, False if it is test set
        :param standardize: Boolean for standardizing the scalar invariants
        :param cap: Caps the max value of the invariants after first normalization pass
        :return: invariants: The num_points X num_scalar_invariants numpy matrix of
                             scalar invariants
        """

        if load is True:
            invars = np.loadtxt("Sij_Rij_gradp_gradk_invars.txt")
            num_invar = invars.shape[1]
        else:
            # Number of invariants = 19 if only one of pressure or tke is True ✓
            num_points = Sij.shape[0]
            num_invar = 19
            if two_invars is True:
                num_invar = 2
            elif pressure is False and tke is False:
                num_invar = 5
            elif pressure is True and tke is True:
                num_invar = 47

            invars = np.full((num_points, num_invar), np.nan)

            # Function for constructing invariants that use the antisymmetric tensors of
            # grad_p and grad_k (cyc perm = cyclic permutation) ✓
            def anti_invars(invars, i, sij, rij, anti, j_start=6):  # ✓
                invars[i, j_start] = tr(dot(anti, anti))
                invars[i, j_start + 1] = tr(mdot([anti, anti, sij]))
                invars[i, j_start + 2] = tr(mdot([anti, anti, sij, sij]))
                invars[i, j_start + 3] = tr(mdot([anti, anti, sij, anti, sij, sij]))
                invars[i, j_start + 4] = tr(dot(rij, anti))
                invars[i, j_start + 5] = tr(mdot([rij, anti, sij]))
                invars[i, j_start + 6] = tr(mdot([rij, anti, sij, sij]))
                invars[i, j_start + 7] = tr(mdot([rij, rij, anti, sij]))  # cyc perm
                invars[i, j_start + 8] = tr(mdot([anti, anti, rij, sij]))
                invars[i, j_start + 9] = tr(mdot([rij, rij, anti, sij, sij]))  # cyc perm
                invars[i, j_start + 10] = tr(mdot([anti, anti, rij, sij, sij]))
                invars[i, j_start + 11] = tr(mdot([rij, rij, sij, anti, sij, sij]))  # cyc
                # perm
                invars[i, j_start + 12] = tr(mdot([anti, anti, sij, rij, sij, sij]))

                return invars

            for i in range(num_points):
                sij = Sij[i, :, :]
                rij = Rij[i, :, :]

                # Invariants of Sij and Rij ✓
                invars[i, 0] = tr(dot(sij, sij))
                invars[i, 1] = tr(dot(rij, rij))
                if two_invars is True:
                    continue

                invars[i, 2] = tr(mdot([sij, sij, sij]))
                invars[i, 3] = tr(mdot([rij, rij, sij]))
                invars[i, 4] = tr(mdot([rij, rij, sij, sij]))

                if pressure is True or tke is True:
                    invars[i, 5] = tr(mdot([rij, rij, sij, rij, sij, sij]))

                # Invariants of Sij, Rij and Ap ✓
                if pressure is True and tke is False:
                    ap = Ap[i, :, :]
                    invars = anti_invars(invars, i, sij, rij, ap)  # ✓

                # Invariants of Sij, Rij and Ak ✓
                elif pressure is False and tke is True:
                    ak = Ak[i, :, :]
                    invars = anti_invars(invars, i, sij, rij, ak)  # ✓

                # Invariants of Sij, Rij, Ap and Ak (cyc perm = cyclic permutation) ✓
                elif pressure is True and tke is True:
                    ap = Ap[i, :, :]
                    invars = anti_invars(invars, i, sij, rij, ap)  # ✓
                    ak = Ak[i, :, :]
                    invars = anti_invars(invars, i, sij, rij, ak, j_start=19)  # ✓

                    invars[i, 32] = tr(dot(ap, ak))
                    invars[i, 33] = tr(mdot([ap, ak, sij]))
                    invars[i, 34] = tr(mdot([ap, ak, sij, sij]))
                    invars[i, 35] = tr(mdot([ap, ap, ak, sij]))  # cyc perm
                    invars[i, 36] = tr(mdot([ak, ak, ap, sij]))
                    invars[i, 37] = tr(mdot([ap, ap, ak, sij, sij]))  # cyc perm
                    invars[i, 38] = tr(mdot([ak, ak, ap, sij, sij]))
                    invars[i, 39] = tr(mdot([ap, ap, sij, ak, sij, sij]))  # cyc perm
                    invars[i, 40] = tr(mdot([ak, ak, sij, ap, sij, sij]))
                    invars[i, 41] = tr(mdot([rij, ap, ak]))
                    invars[i, 42] = tr(mdot([rij, ap, ak, sij]))
                    invars[i, 43] = tr(mdot([rij, ak, ap, sij]))
                    invars[i, 44] = tr(mdot([rij, ap, ak, sij, sij]))
                    invars[i, 45] = tr(mdot([rij, ak, ap, sij, sij]))
                    invars[i, 46] = tr(mdot([rij, ap, sij, ak, sij, sij]))

            # np.savetxt("Sij_Rij_gradp_gradk_invars.txt", invars)

        # Standardize invariants using mean and standard deviation
        # This is recommended over normalization for features that have extremely high or
        # low values ✓
        if standardize is True:
            print("Standardizing invariants with mean and std")
            if self.mu is None or self.std is None:
                is_train = True
            if is_train is True:
                self.mu = np.zeros((num_invar, 2))
                self.std = np.zeros((num_invar, 2))
                self.mu[:, 0] = np.mean(invars, axis=0)  # take mean of all the invariants
                self.std[:, 0] = np.std(invars, axis=0)  # take std of all the invariants

            invars = (invars - self.mu[:, 0]) / self.std[:, 0]
            invars[invars > cap] = cap  # cap max of the invariants
            invars[invars < -cap] = -cap  # cap min of the invariants
            invars = invars * self.std[:, 0] + self.mu[:, 0]  # undo the standardization
            if is_train is True:
                # take mean of all the updated invariants
                self.mu[:, 1] = np.mean(invars, axis=0)
                # take std of all the updated invariants
                self.std[:, 1] = np.std(invars, axis=0)

            # re-standardize a second time after capping
            invars = (invars - self.mu[:, 1]) / self.std[:, 1]
        return invars

    @staticmethod
    def calc_tensor_basis(Sij, Rij, num_tensor_basis, is_scale=True):  # ✓
        """
        Given Sij and Rij, this calculates the tensor basis
        :param Sij: normalized strain rate tensor
        :param Rij: normalized rotation rate tensor
        :param num_tensor_basis: number of tensor bases
        :param is_scale:
        :return: T_flat: num_points X num_tensor_basis X 9 numpy array of tensor basis.
                        Ordering is 11, 12, 13, 21, 22, ...
        """
        num_points = Sij.shape[0]
        T = np.full((num_points, num_tensor_basis, 3, 3), np.nan)

        # Insert tensor basis functions into a dictionary ✓
        def t1(T, i, sij, rij):  # ✓
            T[i, 0, :, :] = sij
            return T

        def t2(T, i, sij, rij):  # ✓
            T[i, 1, :, :] = np.dot(sij, rij) - np.dot(rij, sij)
            return T

        def t3(T, i, sij, rij):  # ✓
            T[i, 2, :, :] = np.dot(sij, sij) - ((1/3) * np.eye(3) *
                                                np.trace(np.dot(sij, sij)))
            return T

        def t4(T, i, sij, rij):  # ✓
            T[i, 3, :, :] = np.dot(rij, rij) - ((1/3) * np.eye(3) *
                                                np.trace(np.dot(rij, rij)))
            return T

        def t5(T, i, sij, rij):  # ✓
            T[i, 4, :, :] = np.dot(rij, np.dot(sij, sij)) - np.dot(np.dot(sij, sij), rij)
            return T

        def t6(T, i, sij, rij):  # ✓
            T[i, 5, :, :] = np.dot(rij, np.dot(rij, sij)) + \
                            np.dot(sij, np.dot(rij, rij)) \
                            - ((2/3) * np.eye(3) *
                               np.trace(np.dot(sij, np.dot(rij, rij))))
            return T

        def t7(T, i, sij, rij):  # ✓
            T[i, 6, :, :] = np.dot(np.dot(rij, sij), np.dot(rij, rij)) - \
                            np.dot(np.dot(rij, rij), np.dot(sij, rij))
            return T

        def t8(T, i, sij, rij):  # ✓
            T[i, 7, :, :] = np.dot(np.dot(sij, rij), np.dot(sij, sij)) - \
                            np.dot(np.dot(sij, sij), np.dot(rij, sij))
            return T

        def t9(T, i, sij, rij):  # ✓
            T[i, 8, :, :] = np.dot(np.dot(rij, rij), np.dot(sij, sij)) + \
                            np.dot(np.dot(sij, sij), np.dot(rij, rij)) \
                            - ((2/3) * np.eye(3) * np.trace(np.dot(np.dot(sij, sij),
                                                                   np.dot(rij, rij))))
            return T

        def t10(T, i, sij, rij):  # ✓
            T[i, 9, :, :] = np.dot(np.dot(rij, np.dot(sij, sij)), np.dot(rij, rij)) \
                            - np.dot(np.dot(rij, np.dot(rij, sij)), np.dot(sij, rij))
            return T

        if num_tensor_basis == 3:
            tb_funcs = [t1, t2, t3]
        elif num_tensor_basis == 10:
            tb_funcs = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]
        else:
            raise Exception('Invalid number of tensor bases')

        tensor_basis_dict = {}
        for j, t in enumerate(tb_funcs):
            tensor_basis_dict[j] = t

        for i in range(num_points):
            sij = Sij[i, :, :]
            rij = Rij[i, :, :]
            for j in range(num_tensor_basis):
                T = tensor_basis_dict[j](T, i, sij=sij, rij=rij)
                # Enforce zero trace for anisotropy
                T[i, j, :, :] = T[i, j, :, :] - ((1/3)*np.eye(3)*np.trace(T[i, j, :, :]))

        # Scale down to promote convergence ✓
        if is_scale:
            scale_factor = [10, 100, 100, 100, 1000, 1000, 10000, 10000, 10000, 10000]
            for j in range(num_tensor_basis):
                T[:, j, :, :] /= scale_factor[j]

        # Flatten T array [CHECK IN CODERUN]
        T_flat = np.full((num_points, num_tensor_basis, 9), np.nan)
        for i in range(3):
            for j in range(3):
                T_flat[:, :, (3*i)+j] = T[:, :, i, j]
        return T_flat

    @staticmethod
    def calc_true_output(tauij, output_var="bij"):
        """
        Given Reynolds stress tensor (num_points X 3 X 3), return flattened
        non-dimensional output tensor to use as true values.
        :param tauij: Reynolds stress tensor
        :param output_var: true output variable name
        :return: flat non-dimensional true output tensor in the form (num_points X 9)
        """

        assert output_var in ["nd_tauij", "bij"]
        num_points = tauij.shape[0]
        output = np.full((num_points, 9), np.nan)
        tke = 0.5 * (tauij[:, 0, 0] + tauij[:, 1, 1] + tauij[:, 2, 2])
        tke = np.maximum(tke, 1e-8)

        # Populate flattened output array
        for i in range(3):
            for j in range(3):
                output[:, (3*i)+j] = tauij[:, i, j]/(2 * tke)

        # Return non-dimensional deviatoric tensor if output is anisotropy bij
        if output_var == "bij":
            for col in [0, 4, 8]:
                output[:, col] -= 1/3

        return output

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
    def make_realizable(labels):  # [UNCHECKED]
        """
        Given the anisotropy tensor, this function forces realizability by shifting
        values within acceptable ranges for Aii > -1/3 and 2|Aij| < Aii + Ajj + 2/3.
        Then, if eigenvalues negative, shifts them to zero. Noteworthy that this step can
        undo constraints from first step, so this function should be called iteratively
        to get convergence to a realizable state.
        :param labels: Flattened anisotropy tensor bij (num_points X 9 array)
        """
        num_points = labels.shape[0]
        A = np.zeros((3, 3))
        for i in range(num_points):
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
