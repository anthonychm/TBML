"""
This file contains functions for creating the 3D non-unique mapping (NUM) grid.
This is a plot that segments the tr(S^2) vs. tr(R^2) vs. supp_var input parameter space
into a 3D grid and shows the number of one-to-many relations between the inputs and
optimal g_n coefficients in each grid cell.

python 3.10 (Asus)
matplotlib 3.7.1 (Asus)
numpy 1.23.5 (Asus)
scikit-learn 1.2.2 (Asus)
seaborn 0.12.2 (Asus)
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sb
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


class ClusterClass3d:
    def __init__(self, S_sq_trace, R_sq_trace, supp_var, g_coeffs, case, g_coeff_name):
        self.S_sq_trace = S_sq_trace
        self.R_sq_trace = R_sq_trace
        self.supp_var = supp_var
        self.g_coeffs = g_coeffs
        self.x_norm = np.nan
        self.y_norm = np.nan
        self.z_norm = np.nan
        self.g_norm = np.nan
        self.case = case
        self.g_coeff_name = g_coeff_name

        if case == "PHLL_case_1p5":
            self.xticks = range(0, 16, 2)
            self.yticks = range(-14, 2, 2)
            self.zticks = np.linspace(0, 0.6, 7)
            self.xlim = [0, 14]
            self.ylim = [-14, 0]
            self.zlim = [0, 0.6]
        elif case == "IMPJ_20000":
            self.xticks = range(0, 14, 2)
            self.yticks = range(-12, 2, 2)
            self.zticks = np.linspace(0, 0.9, 10)
            self.xlim = [0, 12]
            self.ylim = [-12, 0]
            self.zlim = [0, 0.9]
        else:
            raise Exception("Invalid case name")

    def clip_jet_data(self):
        # Remove data points with tr(S²) > 12 and tr(R²) < -12 in IMPJ_20000 data

        assert self.case == "IMPJ_20000"
        S_sq_trace_clipped, R_sq_trace_clipped, supp_var_clipped, g_coeffs_clipped = [], \
            [], [], []
        for i, val in enumerate(self.S_sq_trace):
            if val < 12:
                if self.R_sq_trace[i] > -12:
                    S_sq_trace_clipped.append(val)
                    R_sq_trace_clipped.append(self.R_sq_trace[i])
                    supp_var_clipped.append(self.supp_var[i])
                    g_coeffs_clipped.append(self.g_coeffs[i])
        self.S_sq_trace = S_sq_trace_clipped
        self.R_sq_trace = R_sq_trace_clipped
        self.supp_var = supp_var_clipped
        self.g_coeffs = g_coeffs_clipped

    def min_max_3d_data(self):
        # Define value dictionary for min max normalization
        vdict = {"PHLL_case_1p5": [[-0.1, 0.02], [-0.1, 0], [-0.04, 0.08]],
                 "IMPJ_20000": [[-0.12, 0.02], [-0.1, 0], [-0.04, 0.1]]}

        # Perform min max normalization on x, y, z, and g
        def min_max_func(var, factor=1):
            var_norm = (factor*(var - min(var)))/(max(var) - min(var))
            return var_norm

        self.x_norm = min_max_func(self.S_sq_trace, factor=0.1)
        self.y_norm = min_max_func(self.R_sq_trace, factor=0.1)
        self.z_norm = min_max_func(self.supp_var, factor=0.1)

        g_coeff_int = int([*self.g_coeff_name][1])
        g_min = vdict[self.case][g_coeff_int - 1][0]
        g_max = vdict[self.case][g_coeff_int - 1][1]
        self.g_norm = min_max_func(np.minimum(np.maximum(self.g_coeffs, g_min), g_max))

    def rand_sample(self, n_samples, seed=1):
        # Sample dataset randomly to reduce memory requirements when clustering
        num_points = len(self.x_norm)
        idx = list(range(num_points))
        random.seed(seed)
        random.shuffle(idx)
        sample_idx = idx[:n_samples]
        self.x_norm = self.x_norm[sample_idx]
        self.y_norm = self.y_norm[sample_idx]
        self.z_norm = self.z_norm[sample_idx]
        self.g_norm = self.g_norm[sample_idx]
        self.g_coeffs = np.array(self.g_coeffs)[sample_idx]
        self.S_sq_trace = np.array(self.S_sq_trace)[sample_idx]
        self.R_sq_trace = np.array(self.R_sq_trace)[sample_idx]
        self.supp_var = np.array(self.supp_var)[sample_idx]

    def change_dtype_3d(self):
        self.x_norm = self.x_norm.astype('float32')
        self.y_norm = self.y_norm.astype('float32')
        self.z_norm = self.z_norm.astype('float32')
        self.g_norm = self.g_norm.astype('float32')

    def pre_3d_cluster_fmt(self):
        cluster_data = [[self.x_norm[0], self.y_norm[0], self.z_norm[0], self.g_norm[0]]]
        for i in range(1, len(self.x_norm)):
            cluster_data.append([self.x_norm[i], self.y_norm[i], self.z_norm[i],
                                 self.g_norm[i]])
        cluster_data = np.array(cluster_data)
        return cluster_data

    @staticmethod
    def find_opt_n_clusters(cluster_data, min_clusters, max_clusters,
                            method="complete"):
        sil_avg_list = []
        for n in range(min_clusters, max_clusters + 1):
            clusterer = AgglomerativeClustering(n_clusters=n, linkage=method)
            cluster_labels = clusterer.fit_predict(cluster_data)
            sil_avg = silhouette_score(cluster_data, cluster_labels)
            sil_avg_list.append(sil_avg)
            print("For", n, "clusters, the average silhouette score is: ", sil_avg)
        return sil_avg_list

    @staticmethod
    def plot_all_sil_avg(sil_avg_list1, sil_avg_list2, sil_avg_list3, min_clusters,
                         max_clusters):  # ✓
        """Produce a line plot containing three lines showing how average silhouette
        score changes with the number of clusters for all three g coefficients"""

        plt.figure(figsize=(6, 6))
        plt.plot(range(min_clusters, max_clusters + 1), sil_avg_list1, color="r",
                 marker="o", linestyle="-", label="g₁*", clip_on=False, zorder=10)
        plt.plot(range(min_clusters, max_clusters + 1), sil_avg_list2, color="#00FF00",
                 marker="^", linestyle="--", label="g₂*", clip_on=False, zorder=20)
        plt.plot(range(min_clusters, max_clusters + 1), sil_avg_list3, color="b",
                 marker="s", linestyle="-.", label="g₃*", clip_on=False, zorder=30)
        plt.xlabel("Number of clusters", fontsize=14)
        plt.xticks(range(min_clusters, max_clusters + 2, 2), fontsize=12)
        plt.xlim([2, max_clusters + 0.5])
        plt.ylabel("Average silhouette score", fontsize=14)
        plt.yticks(fontsize=12)
        plt.ylim([0.4, 0.8])
        # plt.legend(fontsize=18)
        # plt.grid(True)
        plt.show()

    def plot_3d_clusters(self, cluster_labels1, cluster_labels2, cluster_labels3,
                         n_clusters1, n_clusters2, n_clusters3):

        # Define function for plotting one cluster subplot
        def plot_cluster_subplot(cluster_labels, n_clusters, ax, title):
            colors = cm.hsv(cluster_labels.astype(float) / n_clusters)
            ax.scatter(self.S_sq_trace, self.R_sq_trace, self.supp_var, c=colors, s=1)
            ax.set(xlim=self.xlim, xticks=self.xticks, ylim=self.ylim,
                   yticks=self.yticks, zlim=self.zlim, zticks=self.zticks, title=title)
            ax.set_xlabel("tr(S²)", fontsize=11)
            ax.set_ylabel("tr(R²)", fontsize=11)
            ax.set_zlabel("Viscosity ratio", fontsize=11)
            ax.view_init(elev=30, azim=-75, roll=0)
            ax.tick_params(axis='both', which='major', labelsize=9)

        # Create subplots
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        # cluster_labels1 -= 1
        # cluster_labels1[cluster_labels1 == -1] = 2
        # cluster_labels1[cluster_labels1 == 0] = 3
        # cluster_labels1[cluster_labels1 == 1] = 0
        # cluster_labels1[cluster_labels1 == 3] = 1
        plot_cluster_subplot(cluster_labels1, n_clusters1, ax1, "g1")

        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        # cluster_labels2 -= 1
        # cluster_labels2[cluster_labels2 == -1] = 2
        plot_cluster_subplot(cluster_labels2, n_clusters2, ax2, "g2")

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        # cluster_labels3 -= 1
        # cluster_labels3[cluster_labels3 == -1] = 2
        # cluster_labels3[cluster_labels3 == 0] = 3
        # cluster_labels3[cluster_labels3 == 1] = 0
        # cluster_labels3[cluster_labels3 == 3] = 1
        plot_cluster_subplot(cluster_labels3, n_clusters3, ax3, "g3")
        plt.show()


def get_3d_cluster_labels(cc, case, n_clusters, find_opt=False):
    cc.min_max_3d_data()
    if case == "IMPJ_20000":
        cc.rand_sample(n_samples=20000)
    cc.change_dtype_3d()
    cluster_data = cc.pre_3d_cluster_fmt()
    if find_opt is True:
        min_clusters, max_clusters = 2, 10
        assert min_clusters > 1
        sil_avg_list = cc.find_opt_n_clusters(cluster_data, min_clusters, max_clusters)
        cluster_labels = False
    else:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete")
        cluster_labels = clusterer.fit_predict(cluster_data)
        min_clusters, max_clusters, sil_avg_list = False, False, False
    return min_clusters, max_clusters, sil_avg_list, cluster_labels


def cluster_3d(S_sq_trace, R_sq_trace, supp_var, g_coeffs, n_clusters, case,
               g_coeff_name):
    cc = ClusterClass3d(S_sq_trace, R_sq_trace, supp_var, g_coeffs, case, g_coeff_name)
    if case == "IMPJ_20000":
        cc.clip_jet_data()
    min_clusters, max_clusters, sil_avg_list, cluster_labels = \
        get_3d_cluster_labels(cc, case, n_clusters)
    return cc, sil_avg_list, min_clusters, max_clusters, cluster_labels


class GridClass3d:
    def __init__(self, case):
        if case == "PHLL_case_1p5":
            self.cell_len = 2
            self.cell_height = 2
            self.cell_depth = 0.1
            self.x_bounds = [0, 14]
            self.y_bounds = [-14, 0]
            self.z_bounds = [0, 0.6]
        elif case == "IMPJ_20000":
            self.cell_len = 2
            self.cell_height = 2
            self.cell_depth = 0.1
            self.x_bounds = [0, 12]
            self.y_bounds = [-12, 0]
            self.z_bounds = [0, 0.9]
        else:
            raise Exception("Invalid case name")

    def create_3d_cells_dict(self, S_sq_trace, R_sq_trace, supp_var):

        # Set number of intervals and cells
        nx = round((self.x_bounds[1] - self.x_bounds[0]) / self.cell_len)
        ny = round((self.y_bounds[1] - self.y_bounds[0]) / self.cell_height)
        nz = round((self.z_bounds[1] - self.z_bounds[0]) / self.cell_depth)
        ncells = nx * ny * nz

        # Calculate x, y, and z intervals
        x_intervals, y_intervals, z_intervals = [], [], []
        for i in range(nx + 1):
            x_intervals.append(self.x_bounds[0] + (self.cell_len * i))

        for i in range(ny + 1):
            y_intervals.append(self.y_bounds[0] + (self.cell_height * i))

        for i in range(nz + 1):
            z_intervals.append(round(self.z_bounds[0] + (self.cell_depth * i), 1))

        # Create cells dictionary
        cells_dict = {i: [] for i in range(ncells)}

        # Append point idx to lists in cells dictionary
        cell_count = -1
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    cell_count += 1
                    cell_len_lower = x_intervals[i]
                    cell_len_upper = x_intervals[i+1]
                    cell_height_lower = y_intervals[j]
                    cell_height_upper = y_intervals[j+1]
                    cell_depth_lower = z_intervals[k]
                    cell_depth_upper = z_intervals[k+1]

                    for idx in range(len(S_sq_trace)):
                        if cell_len_lower < S_sq_trace[idx] < cell_len_upper and \
                                cell_height_lower < R_sq_trace[idx] < cell_height_upper \
                                and cell_depth_lower < supp_var[idx] < cell_depth_upper:
                            cells_dict[cell_count].append(idx)

        return nx, ny, nz, cells_dict

    def create_3d_prox_idx_dict(self, cells_dict, cell_idx, S_sq_trace, R_sq_trace,
                                supp_var):

        prox_idx_dict = {}
        point_idx_list = cells_dict[cell_idx]
        for idx in point_idx_list:
            # Define proximity cuboid limits
            prox_len_lower = S_sq_trace[idx] - (self.cell_len/0.5)
            prox_len_upper = S_sq_trace[idx] + (self.cell_len/0.5)
            prox_height_lower = R_sq_trace[idx] - (self.cell_height/0.5)
            prox_height_upper = R_sq_trace[idx] + (self.cell_height/0.5)
            prox_depth_lower = supp_var[idx] - (self.cell_depth/0.5)
            prox_depth_upper = supp_var[idx] + (self.cell_depth/0.5)
            prox_idx_list = []

            # Fill proximity point index list for each point index
            for prox_idx in range(len(S_sq_trace)):
                if prox_len_lower < S_sq_trace[prox_idx] < prox_len_upper and \
                        prox_height_lower < R_sq_trace[prox_idx] < prox_height_upper and \
                        prox_depth_lower < supp_var[prox_idx] < prox_depth_upper and \
                        idx != prox_idx:
                    prox_idx_list.append(prox_idx)
            prox_idx_dict[idx] = prox_idx_list

        return prox_idx_dict

    @staticmethod
    def create_sph_prox_idx_dict(cells_dict, cell_idx, S_sq_trace, R_sq_trace,
                                 supp_var, r=2):

        prox_idx_dict = {}
        point_idx_list = cells_dict[cell_idx]
        for idx in point_idx_list:
            prox_idx_list = []
            for prox_idx in range(len(S_sq_trace)):
                if np.square(S_sq_trace[prox_idx] - S_sq_trace[idx]) + \
                    np.square(R_sq_trace[prox_idx] - R_sq_trace[idx]) + \
                        np.square(supp_var[prox_idx] - supp_var[idx]) <= np.square(r) \
                        and idx != prox_idx:
                    prox_idx_list.append(prox_idx)
            prox_idx_dict[idx] = prox_idx_list

        return prox_idx_dict

    @staticmethod
    def create_near_n_idx_dict(cells_dict, cell_idx, prox_idx_dict, S_sq_trace,
                               R_sq_trace, supp_var, near_n=3):

        # Get distance to neighbouring points for each idx in the cell
        point_idx_list = cells_dict[cell_idx]
        near_n_idx_dict = {k: [] for k in point_idx_list}
        for idx in point_idx_list:
            assert len(prox_idx_dict[idx]) >= near_n
            dist_list = []
            for adj_idx in prox_idx_dict[idx]:
                dist = np.sqrt(((S_sq_trace[idx] - S_sq_trace[adj_idx])**2) +
                               ((R_sq_trace[idx] - R_sq_trace[adj_idx])**2) +
                               ((supp_var[idx] - supp_var[adj_idx])**2))
                dist_list.append(dist)

            # Find indexes corresponding to nearest n points
            sort_idx = np.argsort(dist_list)
            count = -1
            for i in sort_idx:
                count += 1
                if count < near_n:
                    near_n_idx_dict[idx].append(prox_idx_dict[idx][i])
                else:
                    break

        # Replace prox_idx_dict with near_n_idx_dict for create_flag_nums_dict
        del prox_idx_dict
        prox_idx_dict = near_n_idx_dict

        return prox_idx_dict

    @staticmethod
    def create_flag_nums_dict(cells_dict, cell_idx, prox_idx_dict, cluster_labels):

        point_idx_list = cells_dict[cell_idx]
        flag_nums_dict = {k: 0 for k in point_idx_list}
        for idx in point_idx_list:
            for adj_idx in prox_idx_dict[idx]:
                if cluster_labels[idx] != cluster_labels[adj_idx]:
                    flag_nums_dict[idx] = flag_nums_dict[idx] + 1
        total_nums = sum(flag_nums_dict.values())
        return flag_nums_dict, total_nums

    @staticmethod
    def create_flat_hm_nums_matrix(hm_nums_list, nx, ny, nz):

        annot_3d_array = np.full((nz, ny, nx), np.nan)
        for i, val in enumerate(hm_nums_list):
            z = divmod(i, nx*ny)[0]
            rm = divmod(i, nx*ny)[1]
            y = divmod(rm, nx)[0]
            x = divmod(rm, nx)[1]
            annot_3d_array[z, y, x] = val

        # Sum annot_3d_array in depth direction to collapse into 2D matrix and fill
        # hm_nums_matrix
        annot_matrix = np.sum(annot_3d_array, axis=0)
        hm_nums_matrix = np.full((ny, nx), np.nan)
        for j in range(annot_matrix.shape[0]):
            for i in range(annot_matrix.shape[1]):
                if annot_matrix[j][i] == 0:
                    hm_nums_matrix[j][i] = np.log10(annot_matrix[j][i]+0.1)
                else:
                    hm_nums_matrix[j][i] = np.log10(annot_matrix[j][i])

        # Flip arrays
        annot_matrix = np.flipud(annot_matrix)
        hm_nums_matrix = np.flipud(hm_nums_matrix)
        return annot_matrix, hm_nums_matrix

    @staticmethod
    def plot_flat_hm_subplots(annot_matrix1, hm_nums_matrix1, annot_matrix2,
                              hm_nums_matrix2, annot_matrix3, hm_nums_matrix3):

        # Define function for plotting one heat map subplot
        def plot_hm_subplot(hm_nums_matrix, annot_matrix, ax):
            hm = sb.heatmap(hm_nums_matrix, ax=ax, cmap="rocket_r", annot=annot_matrix,
                            annot_kws={'rotation': 45, 'fontsize': 6}, linewidths=0.5,
                            cbar_kws={'location': 'bottom', 'ticks': range(-1, 6, 1)},
                            vmin=-1, vmax=5, linecolor='black')

            # hm = sb.heatmap(hm_nums_matrix, ax=ax, cmap="rocket_r", linewidths=0.5,
            #                 cbar_kws={'location': 'bottom', 'ticks': range(-1, 6, 1)},
            #                 vmin=-1, vmax=5, linecolor='black')

            hm.set(xticklabels=[], yticklabels=[])
            hm.tick_params(bottom=False, left=False)

        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plot_hm_subplot(hm_nums_matrix1, annot_matrix1, ax1)
        plot_hm_subplot(hm_nums_matrix2, annot_matrix2, ax2)
        plot_hm_subplot(hm_nums_matrix3, annot_matrix3, ax3)
        plt.show()


def calc_flat_hm_nums(cc, cluster_labels, case, nearest_points=True):
    gc = GridClass3d(case)
    nx, ny, nz, cells_dict = gc.create_3d_cells_dict(cc.S_sq_trace, cc.R_sq_trace,
                                                     cc.supp_var)
    hm_nums_list = []
    for i in cells_dict:
        # prox_idx_dict = gc.create_3d_prox_idx_dict(cells_dict, i, cc.S_sq_trace,
        #                                            cc.R_sq_trace, cc.supp_var)
        prox_idx_dict = gc.create_sph_prox_idx_dict(cells_dict, i, cc.S_sq_trace,
                                                    cc.R_sq_trace, cc.supp_var)
        if nearest_points is True:
            prox_idx_dict = gc.create_near_n_idx_dict(cells_dict, i, prox_idx_dict,
                                                      cc.S_sq_trace, cc.R_sq_trace,
                                                      cc.supp_var)
        _, total_nums = gc.create_flag_nums_dict(cells_dict, i, prox_idx_dict,
                                                 cluster_labels)
        hm_nums_list.append(total_nums)
    annot_matrix, hm_nums_matrix = gc.create_flat_hm_nums_matrix(hm_nums_list, nx, ny, nz)
    return gc, annot_matrix, hm_nums_matrix


def plot_3d_subplots(S_sq_trace, R_sq_trace, supp_var, g1, g2, g3, case):
    n_clust1, n_clust2, n_clust3 = 3, 3, 3
    cc1, sil_avg_list1, min_clusters, max_clusters, cluster_labels1 = \
        cluster_3d(S_sq_trace, R_sq_trace, supp_var, g1, n_clust1, case, "g1")
    cc2, sil_avg_list2, min_clusters, max_clusters, cluster_labels2 = \
        cluster_3d(S_sq_trace, R_sq_trace, supp_var, g2, n_clust2, case, "g2")
    cc3, sil_avg_list3, min_clusters, max_clusters, cluster_labels3 = \
        cluster_3d(S_sq_trace, R_sq_trace, supp_var, g3, n_clust3, case, "g3")

    # Check input variables are the same across all clusters ✓
    assert np.array_equal(cc1.S_sq_trace, cc2.S_sq_trace) is True
    assert np.array_equal(cc2.S_sq_trace, cc3.S_sq_trace) is True
    assert np.array_equal(cc1.R_sq_trace, cc2.R_sq_trace) is True
    assert np.array_equal(cc2.R_sq_trace, cc3.R_sq_trace) is True
    assert np.array_equal(cc1.supp_var, cc2.supp_var) is True
    assert np.array_equal(cc2.supp_var, cc3.supp_var) is True

    # Plot average silhouette score for all g coefficients on same plot ✓
    # cc1.plot_all_sil_avg(sil_avg_list1, sil_avg_list2, sil_avg_list3, min_clusters,
    #                      max_clusters)

    # Plot cluster subplots
    # cc1.plot_3d_clusters(cluster_labels1, cluster_labels2, cluster_labels3, n_clust1,
    #                      n_clust2, n_clust3)

    # Calculate NUM instances for heat map plotting all three g coefficients
    gc1, annot_matrix1, hm_nums_matrix1 = calc_flat_hm_nums(cc1, cluster_labels1, case)
    gc2, annot_matrix2, hm_nums_matrix2 = calc_flat_hm_nums(cc2, cluster_labels2, case)
    gc3, annot_matrix3, hm_nums_matrix3 = calc_flat_hm_nums(cc3, cluster_labels3, case)

    # Plot heat map subplots
    gc1.plot_flat_hm_subplots(annot_matrix1, hm_nums_matrix1, annot_matrix2,
                              hm_nums_matrix2, annot_matrix3, hm_nums_matrix3)

    # Save heat map matrices
    np.save('IMPJ_20000_g1_hm_matrix_three_inputs', annot_matrix1)
    np.save('IMPJ_20000_g2_hm_matrix_three_inputs', annot_matrix2)
    np.save('IMPJ_20000_g3_hm_matrix_three_inputs', annot_matrix3)

    print("finish")
