"""
This file contains static methods for calculating heuristic scalars commonly used in
data-driven RANS turbulence modelling.
"""

import numpy as np


class HeuristicCalculator:
    @staticmethod
    def calc_nd_Q(S, R):
        # Calculate non-dim Q-criterion, nd_Q = (||R||^2 - ||S||^2)/(||R||^2 + ||S||^2) ✓
        nd_Q = []
        for row in range(S.shape[0]):
            norm_S = 0
            norm_R = 0
            for i in range(2):
                for j in range(2):
                    norm_S += S[row, i, j] ** 2
                    norm_R += R[row, i, j] ** 2
            norm_S = np.sqrt(norm_S)
            norm_R = np.sqrt(norm_R)
            nd_Q_tmp = ((norm_R ** 2) - (norm_S ** 2)) / ((norm_R ** 2) + (norm_S ** 2))
            nd_Q.append(nd_Q_tmp)
        return nd_Q

    @staticmethod
    def calc_nd_ti(k, Ux, Uy, Uz):
        # Calculate non-dim turbulence intensity, nd_TI = k/(0.5UiUi + k) ✓
        return k / ((0.5 * ((Ux ** 2) + (Uy ** 2) + (Uz ** 2))) + k)

    @staticmethod
    def calc_nd_Ux(Ux, Uy, Uz):
        # Calculate non-dim streamwise velocity, nd_Ux = Ux/mag(U) ✓
        return Ux / np.sqrt((Ux ** 2) + (Uy ** 2) + (Uz ** 2))

    @staticmethod
    def calc_Re_y(k, wall_dist, nu, const=50, min_lim=True, min_lim_var=2):
        # Calculate turbulence Reynolds number, Re_y = sqrt(k)*d/(const*nu) ✓
        Re_y = (np.sqrt(k) * wall_dist) / (const * nu)
        if min_lim is True:  # min(Re_y, min_lim_var)
            Re_y = min(Re_y, min_lim_var)
        return Re_y

    @staticmethod
    def calc_visc_ratio(k, eps, nu, C_mu=0.09):
        # Calculate viscosity ratio, r_nu = nut/((100*nu) + nut) ✓
        nut = C_mu * (k ** 2)/eps
        return nut/((100*nu) + nut)
