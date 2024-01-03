"""
This file contains functions for plotting zonal data after domain partitioning.
matplotlib version 3.7.1 (AM Asus)
"""

import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import trace as tr
from numpy import dot
from numpy.linalg import multi_dot as mdot
import sys
import zonal_marker_calculator as zmc


class ZonalDataPlotter:
    def __init__(self, case_list, zone, zonal_db_dict):
        # Carry out assert checks on the case list and zonal dataset
        assert(len(case_list) == 1)
        assert(zonal_db_dict[zone].shape[1] == 23)
        self.zonal_db_dict = zonal_db_dict

    def calc_tensor_comps(self, zone, limit=True):
        # Calculate tensor components that appear in Pope's 2D GEVH
        dataset = self.zonal_db_dict[zone]
        k_eps = dataset[:, 3]/dataset[:, 4]
        S11 = k_eps*dataset[:, 5]  # S11 = (k/eps)*(du/dx)
        # S12 = 0.5*(k/eps)*[(du/dy)+(dv/dx)]
        S12 = 0.5*k_eps*(dataset[:, 6] + dataset[:, 8])
        # R12 = 0.5*(k/eps)*[(du/dy)-(dv/dx)]
        R12 = 0.5*k_eps*(dataset[:, 6] - dataset[:, 8])

        # Define magnitude limiting function
        def limit_comp(comp, lim_val=1e-10):
            for i in range(len(comp)):
                if -lim_val < comp[i] < 0:
                    comp[i] = -lim_val
                elif 0 <= comp[i] < lim_val:
                    comp[i] = lim_val
            return comp

        # Limit minimum magnitude of the components
        if limit is True:
            S11 = limit_comp(S11)
            S12 = limit_comp(S12)
            R12 = limit_comp(R12)

        return S11, S12, R12

    def calc_norm_s_and_norm_r(self, zone):
        # Calculate norm_s and norm_r
        dataset = self.zonal_db_dict[zone]
        s12 = 0.5*(dataset[:, 6] + dataset[:, 8])
        s13 = 0.5*(dataset[:, 7] + dataset[:, 11])
        s23 = 0.5*(dataset[:, 10] + dataset[:, 12])
        r12 = 0.5*(dataset[:, 6] - dataset[:, 8])
        r13 = 0.5*(dataset[:, 7] - dataset[:, 11])
        r21 = 0.5*(dataset[:, 8] - dataset[:, 6])
        r23 = 0.5*(dataset[:, 10] - dataset[:, 12])
        r31 = 0.5*(dataset[:, 11] - dataset[:, 7])
        r32 = 0.5*(dataset[:, 12] - dataset[:, 10])

        norm_s = np.sqrt((dataset[:, 5]**2) + 2*(s12**2) + 2*(s13**2) +
                         (dataset[:, 9]**2) + 2*(s23**2) + (dataset[:, 13]**2))
        norm_r = np.sqrt(r12**2 + r13**2 + r21**2 + r23**2 + r31**2 + r32**2)
        return norm_s, norm_r

    def calc_timescale_ratio(self, zone, norm_s):
        # Calculate timescale ratio as a supplementary input
        dataset = self.zonal_db_dict[zone]
        k = dataset[:, 3]
        eps = dataset[:, 4]
        t_ratio = (np.multiply(norm_s, k))/(np.multiply(norm_s, k) + eps)
        return t_ratio

    def calc_visc_ratio(self, zone, nu, C_mu):
        # Calculate viscosity ratio as a supplementary input
        dataset = self.zonal_db_dict[zone]
        nut = C_mu*np.divide(np.square(dataset[:, 3]), dataset[:, 4])
        visc_ratio = np.divide(nut, ((100*nu) + nut))
        return visc_ratio

    @staticmethod
    def calc_nondim_qcrit(norm_s, norm_r):
        # Calculate non-dimensional Q-criterion as a supplementary input
        nondim_qcrit = \
            (np.square(norm_r)-np.square(norm_s))/(np.square(norm_r)+np.square(norm_s))
        return nondim_qcrit

    def calc_gradk_gradp_invar(self, zone, case_list):  # ✓
        # Calculate gradk and gradp invariants
        sys.path.insert(1, "../TBNN")
        from TBNN import calculator

        # Reshape gradU
        dataset = self.zonal_db_dict[zone]
        grad_u_flat = dataset[:, 5:14]
        grad_u = np.zeros((grad_u_flat.shape[0], 3, 3))
        for i in range(3):
            for j in range(3):
                grad_u[:, i, j] = grad_u_flat[:, (3*i)+j]

        # Calculate invariants
        pdp = calculator.PopeDataProcessor()
        k = dataset[:, 3]
        eps = dataset[:, 4]
        Sij, Rij = pdp.calc_Sij_Rij(grad_u, k, eps)

        assert len(case_list) == 1
        parent_path = zmc.get_parent_path(case_list[0])

        # Calculate Ak
        def extract_gradk(case_list, parent_path):
            gradk_dict = zmc.load_marker_data_apr2023(case_list[0], parent_path,
                                                      ["gradkx", "gradky", "gradkz"])
            gradk = np.hstack((np.hstack((np.expand_dims(gradk_dict["gradkx"], axis=1),
                                          np.expand_dims(gradk_dict["gradky"], axis=1))),
                               np.expand_dims(gradk_dict["gradkz"], axis=1)))
            return gradk

        gradk = extract_gradk(case_list, parent_path)
        Ak = pdp.calc_Ak(gradk, eps, k)

        # Calculate Ap
        def extract_gradp_and_u(case_list, parent_path):
            gradp_dict = zmc.load_marker_data_apr2023(case_list[0], parent_path,
                                                      ["gradpx", "gradpy", "gradpz",
                                                       "Ux", "Uy", "Uz"])
            gradp = np.hstack((np.hstack((np.expand_dims(gradp_dict["gradpx"], axis=1),
                                          np.expand_dims(gradp_dict["gradpy"], axis=1))),
                               np.expand_dims(gradp_dict["gradpz"], axis=1)))
            u = np.hstack((np.hstack((np.expand_dims(gradp_dict["Ux"], axis=1),
                                      np.expand_dims(gradp_dict["Uy"], axis=1))),
                           np.expand_dims(gradp_dict["Uz"], axis=1)))
            return gradp, u

        gradp, u = extract_gradp_and_u(case_list, parent_path)
        Ap = pdp.calc_Ap(gradp, 1, u, grad_u)

        invar = np.full((dataset.shape[0], 1), np.nan)
        for i in range(dataset.shape[0]):
            sij = Sij[i, :, :]
            rij = Rij[i, :, :]
            ak = Ak[i, :, :]
            ap = Ap[i, :, :]

            invar[i, 0] = tr(mdot([rij, rij, sij, sij]))
            # A = mdot([ak, ap])
            # A_squared = mdot([A, A])
            # invar[i, 0] = 0.5*(((tr(A))**2) - (tr(A_squared)))
        return invar

    def calc_nondim_tau_yy(self, zone, case_list):
        parent_path = zmc.get_parent_path(case_list[0])
        tau_dict = zmc.load_marker_data_apr2023(case_list[0], parent_path, ["tau"])
        tau_yy = tau_dict["tau"][:, 3]
        k = self.zonal_db_dict[zone][:, 3]
        nondim_tau_yy = np.full((k.shape[0], 1), np.nan)
        for i in range(k.shape[0]):
            nondim_tau_yy[i] = tau_yy[i]/k[i]
        return nondim_tau_yy

    def calc_bij_comps(self, zone):
        # Calculate true anisotropy bij components for the dataset
        tauij = self.zonal_db_dict[zone][:, -9:]
        rand_row_idx = random.randint(0, tauij.shape[0])
        assert(tauij[rand_row_idx, 1] == tauij[rand_row_idx, 3])

        k = 0.5*(tauij[:, 0] + tauij[:, 4] + tauij[:, 8])
        b11 = tauij[:, 0]/(2*k) - (1/3)
        b22 = tauij[:, 4]/(2*k) - (1/3)
        b33 = tauij[:, 8]/(2*k) - (1/3)
        b12 = tauij[:, 1]/(2*k)
        return b11, b22, b33, b12

    @staticmethod
    def find_low_or_shear_locs(S_sq_trace, R_sq_trace):
        low_idx_list = []
        shear_idx_list = []
        for i in range(len(S_sq_trace)):
            if 0 < S_sq_trace[i] < 2 and -2 < R_sq_trace[i] < 0:
                low_idx_list.append(i)
            elif -1 < (S_sq_trace[i] + R_sq_trace[i]) < 1 and S_sq_trace[i] < 10 and \
                    R_sq_trace[i] > -10:
                shear_idx_list.append(i)
        return low_idx_list, shear_idx_list

    def plot_scatter_contourf(self, zone, var, var_name, case, low_idx_list,
                              shear_idx_list):

        # Define plot formatting
        xticks = np.linspace(-45, 45, 19) #
        yticks = np.linspace(0, 4, 5) #
        clims_dict = {"S_sq_trace": [0, 12, 7], "R_sq_trace": [-12, 0, 7],
                      "g1": [-0.12, 0.02, 8], "g2": [-0.10, 0.00, 6],
                      "g3": [-0.04, 0.10, 8], "b11": [-0.33, 0.66, 7],
                      "b22": [-0.33, 0.66, 7], "b33": [-0.33, 0.66, 7],
                      "b12": [-0.5, 0.5, 5]}  # format: [min, max, n_intervals] #
        clims = clims_dict[var_name]

        # Create background contour
        Cx = (self.zonal_db_dict[zone][:, 1])/0.1
        Cy = (self.zonal_db_dict[zone][:, 2])/0.1
        cont = plt.tricontourf(Cx, Cy, var, cmap="plasma", extend="both",
                               levels=np.linspace(clims[0], clims[1], (clims[2]*2)-1))
        plt.xlim(min(Cx), max(Cx)) #
        plt.ylim(min(Cy), max(Cy)) #
        plt.xticks(xticks, fontsize=14)
        plt.yticks(yticks, fontsize=14)
        plt.xlabel("x₁/Hₕ", fontsize=18)
        plt.ylabel("x₂/Hₕ", fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        # plt.title(var_name)
        cbar = plt.colorbar(cont)
        cbar.set_ticks(np.round(np.linspace(clims[0], clims[1], clims[2]), decimals=2))
        cbar.set_ticklabels(np.round(np.linspace(clims[0], clims[1], clims[2]),
                                     decimals=2))
        cbar.ax.tick_params(labelsize=14)

        # Gather Cx and Cy values for the low and shear idxs
        # Cx_sc_vals_low = Cx[low_idx_list]
        # Cy_sc_vals_low = Cy[low_idx_list]
        # Cx_sc_vals_shear = Cx[shear_idx_list]
        # Cy_sc_vals_shear = Cy[shear_idx_list]

        # Create foreground scatter points for low and shear idx lists
        # plt.scatter(Cx_sc_vals_low, Cy_sc_vals_low, c='r', s=0.25, marker='o',
        #             alpha=0.5, zorder=2)
        # plt.scatter(Cx_sc_vals_shear, Cy_sc_vals_shear, c='w', s=0.25, marker='o',
        #             alpha=0.7, zorder=3)

        plt.show()

    def create_sorted_contourf_grids(self, zone, var, case):

        # Specify number of discretizations in x and y directions
        if case == "PHLL_case_1p5":
            nx, ny = 99, 149
        else:
            raise Exception("Not a valid case")

        # Specify Cx, Cy and Cx bins
        Cx = self.zonal_db_dict[zone][:, 1]
        Cy = self.zonal_db_dict[zone][:, 2]
        Cx_min, Cx_max = min(Cx), max(Cx)
        Cx_bins = np.linspace(Cx_min, Cx_max, nx)

        # Replace Cx values with their closest Cx bin value
        for count, x in enumerate(Cx):
            idx = (np.abs(Cx_bins - x)).argmin()
            Cx[count] = Cx_bins[idx]

        # Initialise grids
        uniq_Cx = np.unique(Cx)
        assert(len(uniq_Cx) == nx)
        x_grid = np.tile(uniq_Cx, (ny, 1))
        y_grid = np.full((ny, nx), np.NaN)
        z_grid = np.full((ny, nx), np.NaN)

        # Fill unique Cx idx dict
        ux_idx_dict = {ux: [] for ux in uniq_Cx}
        for idx, x in enumerate(Cx):
            ux_idx_dict[x].append(idx)

        # Fill y_grid and z_grid
        for ux_count, ux in enumerate(uniq_Cx):
            Cy_list = Cy[ux_idx_dict[ux]]
            var_list = var[ux_idx_dict[ux]]
            sorted_Cy = np.sort(Cy_list)
            sorted_idx = np.argsort(Cy_list)
            sorted_var = var_list[sorted_idx]

            for y_count, y in enumerate(sorted_Cy):
                y_grid[y_count][ux_count] = y
                z_grid[y_count][ux_count] = sorted_var[y_count]

        return x_grid, y_grid, z_grid

    def create_allocated_contourf_grids(self, zone, var, case):

        # Specify number of discretizations in x and y directions
        if "FBFS" in case:
            nx, ny = 870, 100
        else:
            raise Exception("Not a valid case")

        # Specify unique Cx and Cy
        Cx = self.zonal_db_dict[zone][:, 1]
        Cy = self.zonal_db_dict[zone][:, 2]
        uniq_Cx = np.unique(Cx)
        assert len(uniq_Cx) == nx
        uniq_Cy = np.unique(Cy)
        assert len(uniq_Cy) == ny

        # Initialise grids
        x_grid = np.tile(uniq_Cx, (ny, 1))
        y_grid = np.tile(np.expand_dims(uniq_Cy, axis=1), (1, nx))
        z_grid = np.full((ny, nx), np.NaN)

        # Create dicts with coords as keys and grid indexes as values
        Cx_idx_dict = dict(zip(uniq_Cx, range(uniq_Cx.shape[0])))
        Cy_idx_dict = dict(zip(uniq_Cy, range(uniq_Cy.shape[0])))

        for i in range(Cx.shape[0]):
            xi = Cx_idx_dict[Cx[i]]
            yi = Cy_idx_dict[Cy[i]]
            z_grid[yi, xi] = var[i]

        return x_grid, y_grid, z_grid

    @staticmethod
    def plot_grid_contourf(x_grid, y_grid, z_grid, var_name):

        # Define plot formatting
        xticks = np.linspace(0, 10, 6)
        yticks = np.linspace(0, 3, 7)
        clims_dict = {"S_sq_trace": [0, 14, 8], "R_sq_trace": [-14, 0, 8],
                      "g1": [-0.1, 0.02, 7], "g2": [-0.1, 0, 6],
                      "g3": [-0.04, 0.08, 7], "opt_b11": [-0.33, 0.66, 7],
                      "opt_b22": [-0.33, 0.66, 7], "opt_b33": [-0.33, 0.66, 7],
                      "opt_b12": [-0.5, 0.5, 5], "visc_ratio": [0, 0.9, 10],
                      "other": [-2, 0, 11]}  # format: [min, max, n_intervals] #
        clims = clims_dict[var_name]

        # Create background contour
        cont = plt.contourf(x_grid, y_grid, z_grid, cmap="plasma", extend="both",
                            levels=np.linspace(clims[0], clims[1], (clims[2]*2)-1))
        plt.xlim(np.min(x_grid), np.max(x_grid))
        plt.ylim(np.min(y_grid), np.max(y_grid))
        plt.xticks(xticks, fontsize=14)
        plt.yticks(yticks, fontsize=14)
        # plt.xlabel("x₁/Hₕ", fontsize=18)
        # plt.ylabel("x₂/Hₕ", fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        #plt.title(var_name)
        cbar = plt.colorbar(cont, orientation='horizontal', aspect=40)
        cbar.set_ticks(np.round(np.linspace(clims[0], clims[1], clims[2]), decimals=2))
        cbar.set_ticklabels(np.round(np.linspace(clims[0], clims[1], clims[2]),
                                     decimals=2))
        cbar.ax.tick_params(labelsize=14)
        plt.show()

    @staticmethod
    def plot_grid_default_contourf(x_grid, y_grid, z_grid):
        # xticks = np.linspace(0, 10, 6)
        # yticks = np.linspace(0, 3, 7)
        cont = plt.contourf(x_grid, y_grid, z_grid, cmap="plasma",
                            levels=np.linspace(-0.04, 0.08, 200), extend="both")
        plt.xlim(np.min(x_grid), np.max(x_grid))
        plt.ylim(np.min(y_grid), np.max(y_grid))
        # plt.xticks(xticks, fontsize=14)
        # plt.yticks(yticks, fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        cbar = plt.colorbar(cont, orientation='horizontal', aspect=40,
                            ticks=np.linspace(-0.04, 0.08, 7))
        cbar.ax.tick_params(labelsize=14)
        plt.show()

    def var_scatter(self, zone, var):
        Cx = self.zonal_db_dict[zone][:, 1]
        Cy = self.zonal_db_dict[zone][:, 2]
        plt.scatter(Cx, Cy, c=var, cmap='plasma', s=2)
        plt.colorbar()
        plt.show()

    def var_tricontourf(self, zone, var):
        Cx = self.zonal_db_dict[zone][:, 1]
        Cy = self.zonal_db_dict[zone][:, 2]
        # cont = plt.tricontourf(Cx, Cy, var, cmap="plasma", extend="both")
        cont = plt.tricontourf(Cx, Cy, var, cmap="plasma", extend="both",
                               levels=np.linspace(-0.1, 0.02, 13))
        plt.xlim(min(Cx), max(Cx))
        plt.ylim(min(Cy), max(Cy))
        # plt.xticks(np.linspace(-45, 45, 19))
        # plt.yticks(np.linspace(0, 4, 5))
        cbar = plt.colorbar(cont)
        plt.show()

    @staticmethod
    def calc_traces(S11, S12, R12):
        S_sq_trace = 2*((S11**2) + (S12**2))
        R_sq_trace = -2*(R12**2)
        return S_sq_trace, R_sq_trace

    @staticmethod
    def calc_g1(b11, b22, b12, S11, S12):
        f1 = 1/(2*S11)
        f2 = (S12**2)/(2*S11*((S11**2)+(S12**2)))
        f3 = S12/((S11**2)+(S12**2))
        g1 = ((f1-f2)*(b11-b22)) + (f3*b12)
        return g1

    @staticmethod
    def calc_g2(b11, b22, b12, S11, S12, R12):
        top = (2*S11*b12) + (S12*(b22-b11))
        bottom = 4*R12*((S11**2)+(S12**2))
        g2 = top/bottom
        return g2

    @staticmethod
    def calc_g3(b11, b22, S11, S12):
        g3 = (3*(b11+b22))/(2*((S11**2)+(S12**2)))
        return g3

    @staticmethod
    def calc_opt_g1(b11, b22, b12, S11, S12):
        f1 = S11/(2*((S11**2)+(S12**2)))
        f2 = S12/((S11**2)+(S12**2))
        g1 = (f1*(b11-b22)) + (f2*b12)
        opt_g1 = np.full_like(g1, np.nan)
        for i in range(len(opt_g1)):
            opt_g1[i] = min(g1[i], 0)
        return opt_g1

    @staticmethod
    def calc_opt_g2(opt_g1, b12, S11, S12, R12):
        opt_g2 = (b12-(opt_g1*S12))/(2*S11*R12)
        return opt_g2

    @staticmethod
    def calc_opt_g3(opt_g1, opt_g2, b11, S11, S12, R12):
        opt_g3 = (3*(b11-(opt_g1*S11)+(2*opt_g2*S12*R12)))/((S11**2)+(S12**2))
        return opt_g3

    @staticmethod
    def create_g_coeffs_scatter(g1, g2, g3, S_sq_trace, R_sq_trace):

        def create_subscatter(ax, g, vlims, title, ylabel=None):
            ax.set(xlabel="tr(S²)", ylabel=ylabel, title=title, xlim=[0, 14],
                    xticks=range(0, 16, 2), yticks=range(-14, 2, 2), ylim=[-14, 0])
            ax.tick_params(axis='both', which='major', labelsize=8)
            sc = ax.scatter(S_sq_trace, R_sq_trace, c=g, vmin=vlims[0], vmax=vlims[1],
                            s=0.5, cmap='plasma')
            cbar = ax.figure.colorbar(sc, ax=ax, location="bottom")
            cbar.ax.tick_params(labelsize=8)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        create_subscatter(ax1, g1, [-0.1, 0.02], "g₁", ylabel="tr(R²)")
        ax2 = fig.add_subplot(1, 3, 2)
        create_subscatter(ax2, g2, [-0.1, 0], "g₂")
        ax3 = fig.add_subplot(1, 3, 3)
        create_subscatter(ax3, g3, [-0.04, 0.08], "g₃")
        plt.show()

    @staticmethod
    def create_g_coeffs_3d_scatter(g1, g2, g3, S_sq_trace, R_sq_trace, supp_var):

        def create_subscatter(ax, g, vlims, title):
            ax.set(xlim=[0, 8], xticks=range(0, 10, 2), ylim=[-8, 0],
                   yticks=range(-8, 2, 2), zlim=[0.5, 1],
                   zticks=np.linspace(0.5, 1.1, 7), title=title)
            ax.set_xlabel("tr(S²)", fontsize=11)
            ax.set_ylabel("tr(R²)", fontsize=11)
            ax.set_zlabel("Nondimensional tau_yy", fontsize=11)
            sc = ax.scatter(S_sq_trace, R_sq_trace, supp_var, c=g, s=1, cmap='plasma',
                            vmin=vlims[0], vmax=vlims[1])
            cbar = ax.figure.colorbar(sc, ax=ax, location='bottom')
            cbar.ax.tick_params(labelsize=9)
            ax.view_init(elev=30, azim=-75, roll=0)
            ax.tick_params(axis='both', which='major', labelsize=9)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        create_subscatter(ax1, g1, [-0.1, 0.02], "g1")
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        create_subscatter(ax2, g2, [-0.1, 0], "g2")
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        create_subscatter(ax3, g3, [-0.04, 0.08], "g3")
        plt.show()

    @staticmethod
    def create_bij_scatter(b11, b22, b33, b12, S_sq_trace, R_sq_trace):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

        # b11 subplot
        ax1.set(xlabel="tr(S²)", ylabel="tr(R²)", title="b₁₁", xlim=[0, 14],
                ylim=[-14, 0], xticks=range(0, 16, 2))
        ax1.tick_params(axis='both', which='major', labelsize=8)
        s1 = ax1.scatter(S_sq_trace, R_sq_trace, c=b11, vmin=0,
                         vmax=0.2, s=8, cmap='plasma')
        cbar1 = ax1.figure.colorbar(s1, ax=ax1, location="bottom")
        cbar1.ax.tick_params(labelsize=8)

        # b22 subplot
        ax2.set(xlabel="tr(S²)", title="b₂₂", xlim=[0, 14], ylim=[-14, 0],
                xticks=range(0, 16, 2))
        ax2.tick_params(axis='both', which='major', labelsize=8)
        s2 = ax2.scatter(S_sq_trace, R_sq_trace, c=b22, vmin=-0.2,
                         vmax=0, s=8, cmap='plasma')
        cbar2 = ax2.figure.colorbar(s2, ax=ax2, location="bottom")
        cbar2.ax.tick_params(labelsize=8)

        # b33 subplot
        ax3.set(xlabel="tr(S²)", title="b₃₃", xlim=[0, 14], ylim=[-14, 0],
                xticks=range(0, 16, 2))
        ax3.tick_params(axis='both', which='major', labelsize=8)
        s3 = ax3.scatter(S_sq_trace, R_sq_trace, c=b33, vmin=-0.1,
                         vmax=0.15, s=8, cmap='plasma')
        cbar3 = ax3.figure.colorbar(s3, ax=ax3, location="bottom")
        cbar3.ax.tick_params(labelsize=8)

        # b12 subplot
        ax4.set(xlabel="tr(S²)", title="b₁₂", xlim=[0, 14], ylim=[-14, 0],
                xticks=range(0, 16, 2))
        ax4.tick_params(axis='both', which='major', labelsize=8)
        s4 = ax4.scatter(S_sq_trace, R_sq_trace, c=b12, vmin=-0.2,
                         vmax=0.05, s=8, cmap='plasma')
        cbar4 = ax4.figure.colorbar(s4, ax=ax4, location="bottom")
        cbar4.ax.tick_params(labelsize=8)

        plt.show()
