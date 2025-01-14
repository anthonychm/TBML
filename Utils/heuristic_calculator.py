"""
This file contains static methods for calculating heuristic scalars commonly used in
data-driven RANS turbulence modelling.
"""

import numpy as np
from Utils.reformatter import Reformatter


class VariableCalculator:
    @staticmethod
    def calc_frob_norm(var):
        norm_var = np.zeros(len(var, ))
        for i in range(3):
            for j in range(3):
                norm_var += var[:, i, j] ** 2
        return np.sqrt(norm_var)

    @staticmethod
    def calc_bh_tauij(k, eps, S, C_mu=0.09):
        # Calculate Reynolds stress based on Boussinesq Hypothesis
        nut = C_mu*(k ** 2)/eps
        nut = np.expand_dims(nut, axis=(1, 2))
        tauij = 2*nut*S
        for i in range(3):
            for j in range(3):
                if i == j:
                    tauij[:, i, j] -= (2/3)*k
        return tauij

    @staticmethod
    def calc_S_and_R(gradU):
        # Calculate dimensional mean strain rate S and mean rotation rate R, dim = [1/s] ✓
        if gradU.ndim == 2:
            gradU = Reformatter.flat_tensor_to_tensor(gradU)
        S = np.full((len(gradU), 3, 3), np.nan)
        R = np.full((len(gradU), 3, 3), np.nan)
        for i in range(len(gradU)):
            S[i, :, :] = 0.5 * (gradU[i, :, :] + np.transpose(gradU[i, :, :]))
            R[i, :, :] = 0.5 * (gradU[i, :, :] - np.transpose(gradU[i, :, :]))
        return S, R


class HeuristicCalculator:
    @staticmethod
    def calc_nd_Q(S, R):
        # Calculate non-dim Q-criterion, nd_Q = (||R||^2 - ||S||^2)/(||R||^2 + ||S||^2)
        norm_S = VariableCalculator.calc_frob_norm(S)
        norm_R = VariableCalculator.calc_frob_norm(R)
        return ((norm_R ** 2) - (norm_S ** 2)) / ((norm_R ** 2) + (norm_S ** 2))

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
            Re_y = np.expand_dims(Re_y, axis=1)
            min_lim_col = np.full((len(Re_y), 1), min_lim_var)
            Re_y = np.hstack((Re_y, min_lim_col))
            Re_y = np.min(Re_y, axis=1)
        return Re_y

    @staticmethod
    def calc_visc_ratio(k, eps, nu, C_mu=0.09):
        # Calculate viscosity ratio, r_nu = nut/((100*nu) + nut) ✓
        nut = C_mu * (k ** 2)/eps
        return nut/((100*nu) + nut)
