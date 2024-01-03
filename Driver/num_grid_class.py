"""
This file contains functions for creating the non-unique mapping (NUM) grid. The NUM
grid is a plot that segments the tr(S^2) inputs vs. tr(R^2) inputs parameter space
into a grid and illustrates the number of one-to-many relations between the inputs and
optimal g_n coefficients in each grid cell.

python 3.10
matplotlib version 3.7.1 (Asus)
numpy version 1.23.5 (Asus)
scikit-learn version 1.2.2 (Asus)
seaborn version 0.12.2 (Asus)
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sb
import timeit
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


class ClusterClass:
    def __init__(self, S_sq_trace, R_sq_trace, g_coeffs, case, g_coeff_name):  # ✓
        self.S_sq_trace = S_sq_trace
        self.R_sq_trace = R_sq_trace
        self.g_coeffs = g_coeffs
        self.x_norm = np.nan
        self.y_norm = np.nan
        self.g_norm = np.nan
        self.case = case
        self.g_coeff_name = g_coeff_name

        if case == "PHLL_case_1p5":
            self.xticks = range(0, 16, 2)
            self.yticks = range(-14, 2, 2)
            self.xlim = [0, 14]
            self.ylim = [-14, 0]
        elif case == "BUMP_h38":
            self.xticks = range(0, 600, 100)
            self.yticks = range(-21, 3, 3)
            self.xlim = [0, 500]
            self.ylim = [-21, 0]
        elif case == "IMPJ_20000":
            self.xticks = range(0, 14, 2)
            self.yticks = range(-12, 2, 2)
            self.xlim = [0, 12]
            self.ylim = [-12, 0]
        else:
            raise Exception("Invalid case name")

    def clip_bump_data(self):  # ✓
        """Remove data points with tr(S²) > 70 in BUMP_h38 data"""

        assert self.case == "BUMP_h38"
        S_sq_trace_clipped, R_sq_trace_clipped, g_coeffs_clipped = [], [], []
        for i, val in enumerate(self.S_sq_trace):
            if val < 70:
                S_sq_trace_clipped.append(val)
                R_sq_trace_clipped.append(self.R_sq_trace[i])
                g_coeffs_clipped.append(self.g_coeffs[i])
        self.S_sq_trace = S_sq_trace_clipped
        self.R_sq_trace = R_sq_trace_clipped
        self.g_coeffs = g_coeffs_clipped

    def clip_jet_data(self):  # ✓
        """Remove data points with tr(S²) > 12 and tr(R²) < -12 in IMPJ_20000 data"""

        assert self.case == "IMPJ_20000"
        S_sq_trace_clipped, R_sq_trace_clipped, g_coeffs_clipped = [], [], []
        for i, val in enumerate(self.S_sq_trace):
            if val < 12:
                if self.R_sq_trace[i] > -12:
                    S_sq_trace_clipped.append(val)
                    R_sq_trace_clipped.append(self.R_sq_trace[i])
                    g_coeffs_clipped.append(self.g_coeffs[i])
        self.S_sq_trace = S_sq_trace_clipped
        self.R_sq_trace = R_sq_trace_clipped
        self.g_coeffs = g_coeffs_clipped

    def min_max_data(self):  # ✓
        """Non-dimensionalizes all data for clustering with min-max normalization"""

        # Define value dictionary for min max normalization
        vdict = {"PHLL_case_1p5": [[-0.1, 0.02], [-0.1, 0], [-0.04, 0.08]],
                 "BUMP_h38": [[-0.12, 0.02], [-0.1, 0.02], [-0.04, 0.1]],
                 "IMPJ_20000": [[-0.12, 0.02], [-0.1, 0], [-0.04, 0.1]]}

        # Perform min max normalization on x, y and g
        def min_max_func(var, factor=1):
            var_norm = (factor*(var - min(var)))/(max(var) - min(var))
            return var_norm

        self.x_norm = min_max_func(self.S_sq_trace, factor=0.1)
        self.y_norm = min_max_func(self.R_sq_trace, factor=0.1)

        g_coeff_int = int([*self.g_coeff_name][1])
        g_min = vdict[self.case][g_coeff_int-1][0]
        g_max = vdict[self.case][g_coeff_int-1][1]
        self.g_norm = min_max_func(np.minimum(np.maximum(self.g_coeffs, g_min), g_max))

    def rand_sample(self, n_samples, seed=1):  # ✓
        """Sample dataset randomly to reduce memory requirements when clustering"""

        num_points = len(self.x_norm)
        idx = list(range(num_points))
        random.seed(seed)
        random.shuffle(idx)
        sample_idx = idx[:n_samples]
        self.x_norm = self.x_norm[sample_idx]
        self.y_norm = self.y_norm[sample_idx]
        self.g_norm = self.g_norm[sample_idx]
        self.g_coeffs = np.array(self.g_coeffs)[sample_idx]
        self.S_sq_trace = np.array(self.S_sq_trace)[sample_idx]
        self.R_sq_trace = np.array(self.R_sq_trace)[sample_idx]

    def change_dtype(self):  # ✓
        """Change data type from float64 to float32 to reduce memory requirements"""

        self.x_norm = self.x_norm.astype('float32')
        self.y_norm = self.y_norm.astype('float32')
        self.g_norm = self.g_norm.astype('float32')

    def pre_cluster_format(self):  # ✓
        """Format data for clustering"""

        cluster_data = [[self.x_norm[0], self.y_norm[0], self.g_norm[0]]]
        for i in range(1, len(self.x_norm)):
            cluster_data.append([self.x_norm[i], self.y_norm[i], self.g_norm[i]])
        cluster_data = np.array(cluster_data)
        return cluster_data

    @staticmethod
    def find_opt_n_clusters(cluster_data, min_clusters, max_clusters,
                            method="complete"):  # ✓
        """Cluster all data points according to x_norm, y_norm and g_norm values. Find
        optimal number of clusters using silhouette score."""

        sil_avg_list = []
        for n in range(min_clusters, max_clusters+1):
            clusterer = AgglomerativeClustering(n_clusters=n, linkage=method)
            cluster_labels = clusterer.fit_predict(cluster_data)
            sil_avg = silhouette_score(cluster_data, cluster_labels)
            sil_avg_list.append(sil_avg)
            print("For", n, "clusters, the average silhouette score is: ", sil_avg)
        return sil_avg_list

    @staticmethod
    def plot_sil_avg(sil_avg_list, min_clusters, max_clusters):  # ✓
        """Produce a line plot showing how average silhouette score changes with the
        number of clusters"""

        plt.plot(range(min_clusters, max_clusters+1), sil_avg_list, 'o-b')
        plt.xlabel("Number of clusters", fontsize=8)
        plt.xticks(range(min_clusters, max_clusters+2, 2), fontsize=8)
        plt.ylabel("Average silhouette score", fontsize=8)
        plt.yticks(fontsize=8)
        plt.show()

    @staticmethod
    def plot_all_sil_avg(sil_avg_list1, sil_avg_list2, sil_avg_list3, min_clusters,
                         max_clusters, avg_lists=True):  # ✓
        """Produce a line plot containing three lines showing how average silhouette
        score changes with the number of clusters for all three g coefficients"""

        plt.figure(figsize=(6, 6))
        plt.plot(range(min_clusters, max_clusters+1), sil_avg_list1, color="r",
                 marker="o", linestyle="--", label="true g₁", clip_on=False, zorder=10,
                 markersize=10)
        plt.plot(range(min_clusters, max_clusters+1), sil_avg_list2, color="#00FF00",
                 marker="^", linestyle="-.", label="true g₂", clip_on=False, zorder=20,
                 markersize=10)
        plt.plot(range(min_clusters, max_clusters+1), sil_avg_list3, color="b",
                 marker="s", linestyle=":", label="true g₃", clip_on=False, zorder=30,
                 markersize=10)

        if avg_lists is True:
            sil_avg_array = np.vstack((np.array(sil_avg_list1), np.array(sil_avg_list2),
                                       np.array(sil_avg_list3)))
            sil_avg_avg = np.mean(sil_avg_array, axis=0)
            plt.plot(range(min_clusters, max_clusters+1), sil_avg_avg, color="k",
                     marker="D", linestyle="-", label="average over true gₙ",
                     clip_on=False, zorder=40, markersize=10)

        # plt.xlabel("Number of clusters", fontsize=14)
        plt.xticks(range(min_clusters, max_clusters+2, 2), fontsize=12)
        plt.xlim([2, max_clusters+0.5])
        # plt.ylabel("Average silhouette score", fontsize=14)
        plt.yticks(fontsize=12)
        plt.ylim([0.45, 0.75])
        # plt.legend(fontsize=18)
        # plt.grid(True)
        plt.show()

    def plot_clusters(self, cluster_labels, n_clusters, g_coeff_name):  # ✓
        """Plot clusters representing smooth mappings between inputs tr(S²) and tr(R²)
        and output g coefficients"""

        colors = cm.hsv(cluster_labels.astype(float)/n_clusters)
        plt.scatter(self.S_sq_trace, self.R_sq_trace, c=colors, s=0.5)
        plt.xlabel("tr(S²)")
        plt.ylabel("tr(R²)")
        plt.xticks(self.xticks)
        plt.yticks(self.yticks)
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.title(g_coeff_name)
        plt.show()

    def plot_clusters_subplots(self, cluster_labels1, cluster_labels2, cluster_labels3,
                               n_clusters1, n_clusters2, n_clusters3):  # ✓
        """Plot subplots of clusters representing smooth mappings between inputs tr(S²)
        and tr(R²) and output g coefficients for g1, g2 and g3"""

        # Define function for plotting one cluster subplot
        def plot_cluster_subplot(cluster_labels, n_clusters, ax, title, ylabel=None):
            colors = cm.hsv(cluster_labels.astype(float) / n_clusters)
            ax.scatter(self.S_sq_trace, self.R_sq_trace, c=colors, s=10)
            ax.set_xlabel("tr(S²)", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set(title=title, xticks=self.xticks, yticks=self.yticks, xlim=self.xlim,
                   ylim=self.ylim)
            ax.tick_params(axis='both', which='major', labelsize=10)

        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plot_cluster_subplot(cluster_labels1, n_clusters1, ax1, "g₁", ylabel="tr(R²)")
        plot_cluster_subplot(cluster_labels2, n_clusters2, ax2, "g₂")
        plot_cluster_subplot(cluster_labels3, n_clusters3, ax3, "g₃")
        plt.show()

    def plot_clusters_sep(self, cluster_labels1, cluster_labels2, cluster_labels3,
                          n_clusters1, n_clusters2, n_clusters3, g1, g2, g3, case):  # ✓
        """Create subplots showing the separate clusters and their tr(S²) and tr(R²)
        values"""

        assert n_clusters1 == n_clusters2 == n_clusters3
        n_clusters = n_clusters1
        fig, axs = plt.subplots(nrows=n_clusters, ncols=3)

        # Define value dictionary for scatter plot vmin and vmax
        vdict = {"PHLL_case_1p5": [[-0.1, 0.02], [-0.1, 0], [-0.04, 0.08]],
                 "BUMP_h38": [[-0.12, 0.02], [-0.1, 0.02], [-0.04, 0.1]],
                 "IMPJ_20000": [[-0.12, 0.02], [-0.1, 0], [-0.04, 0.1]]}

        # Extract input and output data for a subplot
        def get_io_sep(cluster_labels, label, g_coeff):
            x, y, g = [], [], []
            for count, val in enumerate(cluster_labels):
                if val == label:
                    x.append(self.S_sq_trace[count])
                    y.append(self.R_sq_trace[count])
                    g.append(g_coeff[count])
            return x, y, g

        # Create subplots
        for i in range(3):
            for j in range(n_clusters):
                x, y, g = get_io_sep(locals()['cluster_labels' + str(i+1)], j,
                                     locals()['g' + str(i+1)])
                sc = axs[j, i].scatter(x, y, c=g, s=0.1, vmin=vdict[case][i][0],
                                       vmax=vdict[case][i][1], cmap='plasma')
                axs[j, i].set(xticks=self.xticks, yticks=self.yticks, xlim=self.xlim,
                              ylim=self.ylim)
                axs[j, i].tick_params(axis='both', which='major', labelsize=6)
                if j == 0:
                    cbar = axs[j, i].figure.colorbar(sc, ax=axs[j, i], location="top")
                    cbar.ax.tick_params(labelsize=6)
                if j == n_clusters-1:
                    axs[j, i].set(xlabel="tr(S²)")
                if i == 0:
                    axs[j, i].set(ylabel="tr(R²)")
        plt.show()


class GridClass:
    def __init__(self, case):  # ✓
        if case == "PHLL_case_1p5":
            self.cell_len = 2
            self.cell_height = 2
            self.x_bounds = [0, 14]
            self.y_bounds = [-14, 0]
        elif case == "BUMP_h38":
            self.cell_len = 100
            self.cell_height = 3
            self.x_bounds = [0, 500]
            self.y_bounds = [-21, 0]
        elif case == "IMPJ_20000":
            self.cell_len = 2
            self.cell_height = 2
            self.x_bounds = [0, 12]
            self.y_bounds = [-12, 0]
        else:
            raise Exception("Invalid case name")

    def create_cells_dict(self, S_sq_trace, R_sq_trace):  # ✓
        """
        Creates dictionary with cells as keys and a list of point indexes that are
        contained in those cells as values.
        {cell (int) = [list of indexes (int) corresponding to enclosed points]}
        x = S_sq_trace
        y = R_sq_trace
        cell = grid rectangle
        """

        # Set number of length and height intervals, and number of cells
        nx = round((self.x_bounds[1] - self.x_bounds[0])/self.cell_len)
        ny = round((self.y_bounds[1] - self.y_bounds[0])/self.cell_height)
        ncells = nx*ny

        # Calculate x and y intervals
        x_intervals, y_intervals = [], []
        for i in range(nx+1):
            x_intervals.append(self.x_bounds[0] + (self.cell_len*i))

        for i in range(ny+1):
            y_intervals.append(self.y_bounds[0] + (self.cell_height*i))

        # Create cells dictionary
        cells_dict = {i: [] for i in range(ncells)}

        # Append point idx to lists in cells dictionary
        cell_count = -1
        for j in range(ny):
            for i in range(nx):
                cell_count += 1
                cell_len_lower = x_intervals[i]
                cell_len_upper = x_intervals[i+1]
                cell_height_lower = y_intervals[j]
                cell_height_upper = y_intervals[j+1]

                for idx in range(len(S_sq_trace)):
                    if cell_len_lower < S_sq_trace[idx] < cell_len_upper and \
                            cell_height_lower < R_sq_trace[idx] < cell_height_upper:
                        cells_dict[cell_count].append(idx)

        return nx, ny, cells_dict

    def create_prox_idx_dict(self, cells_dict, cell_idx, S_sq_trace, R_sq_trace):  # ✓
        """ For a cell:
        Creates dictionary with point indexes in the cell as keys and a list of indexes of
        neighbouring points as values.
        {point index (int) = [list of point indexes (int) corresponding to neighbouring
        points]} """

        prox_idx_dict = {}
        point_idx_list = cells_dict[cell_idx]
        for idx in point_idx_list:
            # Define proximity square limits
            prox_len_lower = S_sq_trace[idx] - (self.cell_len/0.625)
            prox_len_upper = S_sq_trace[idx] + (self.cell_len/0.625)
            prox_height_lower = R_sq_trace[idx] - (self.cell_height/0.625)
            prox_height_upper = R_sq_trace[idx] + (self.cell_height/0.625)
            prox_idx_list = []

            # Fill proximity point index list for each point index
            for prox_idx in range(len(S_sq_trace)):
                if prox_len_lower < S_sq_trace[prox_idx] < prox_len_upper and \
                        prox_height_lower < R_sq_trace[prox_idx] < prox_height_upper \
                        and idx != prox_idx:
                    prox_idx_list.append(prox_idx)
            prox_idx_dict[idx] = prox_idx_list

        return prox_idx_dict

    @staticmethod
    def create_circ_prox_idx_dict(cells_dict, cell_idx, S_sq_trace, R_sq_trace, r=2):
        """ For a cell:
        Creates dictionary with point indexes in the cell as keys and a list of indexes
        of points within their circle of vicinity as values.
        {point index (int) = [list of point indexes (int) corresponding to neighbouring
        points]} """

        prox_idx_dict = {}
        point_idx_list = cells_dict[cell_idx]
        for idx in point_idx_list:
            prox_idx_list = []
            for prox_idx in range(len(S_sq_trace)):
                if np.square(S_sq_trace[prox_idx] - S_sq_trace[idx]) + \
                    np.square(R_sq_trace[prox_idx] - R_sq_trace[idx]) <= np.square(r) \
                        and idx != prox_idx:
                    prox_idx_list.append(prox_idx)
            prox_idx_dict[idx] = prox_idx_list

        return prox_idx_dict

    @staticmethod
    def create_near_n_idx_dict(cells_dict, cell_idx, prox_idx_dict, S_sq_trace,
                               R_sq_trace, near_n=3):  # ✓
        """For a cell:
        Creates a dictionary with point indexes in the cells as keys and a list of
        indexes corresponding to the nearest n number of points as values.
        {point index (int) = [list of point indexes (int) corresponding to nearest n
        number of points]}"""

        # Get distance to neighbouring points for each idx in the cell
        point_idx_list = cells_dict[cell_idx]
        near_n_idx_dict = {k: [] for k in point_idx_list}
        for idx in point_idx_list:
            assert len(prox_idx_dict[idx]) >= near_n
            dist_list = []
            for adj_idx in prox_idx_dict[idx]:
                dist = np.sqrt(((S_sq_trace[idx] - S_sq_trace[adj_idx])**2) +
                               ((R_sq_trace[idx] - R_sq_trace[adj_idx])**2))
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
    def create_flag_nums_dict(cells_dict, cell_idx, prox_idx_dict, cluster_labels):  # ✓
        """For a cell:
        Creates dictionary with point indexes in the cells as keys and a binary list
        corresponding to whether its neighbouring points are in its mapping or not as
        values.
        {point index (int) = [binary list (int) where each element corresponds to a
        neighbouring point]}"""

        point_idx_list = cells_dict[cell_idx]
        flag_nums_dict = {k: 0 for k in point_idx_list}
        for idx in point_idx_list:
            for adj_idx in prox_idx_dict[idx]:
                if cluster_labels[idx] != cluster_labels[adj_idx]:
                    flag_nums_dict[idx] = flag_nums_dict[idx] + 1
        total_nums = sum(flag_nums_dict.values())
        return flag_nums_dict, total_nums

    @staticmethod
    def create_hm_nums_matrix(hm_nums_list, nx, ny):  # ✓
        """Fill annot_matrix and rearrange heat_map_nums_list into heat_map_nums_matrix"""

        annot_matrix = np.full((ny, nx), np.nan)
        hm_nums_matrix = np.full((ny, nx), np.nan)
        for i, val in enumerate(hm_nums_list):
            y = divmod(i, nx)[0]
            x = divmod(i, nx)[1]
            annot_matrix[y, x] = val
            if val == 0:
                hm_nums_matrix[y, x] = np.log10(val+0.1)
            else:
                hm_nums_matrix[y, x] = np.log10(val)
        annot_matrix = np.flipud(annot_matrix)
        hm_nums_matrix = np.flipud(hm_nums_matrix)
        return annot_matrix, hm_nums_matrix

    @staticmethod
    def plot_heat_map(annot_matrix, hm_nums_matrix):  # ✓
        """Create and format a single heat map"""

        hm = sb.heatmap(hm_nums_matrix, cmap="rocket_r", annot=annot_matrix,
                        annot_kws={'rotation': 45}, linewidths=0.5, linecolor='black',
                        cbar_kws={'location': 'bottom'}, vmin=-1, vmax=7)
        hm.set(xticklabels=[], yticklabels=[])
        hm.tick_params(bottom=False, left=False)
        plt.show()

    @staticmethod
    def plot_heat_map_subplots(annot_matrix1, hm_nums_matrix1, annot_matrix2,
                               hm_nums_matrix2, annot_matrix3, hm_nums_matrix3):  # ✓
        """Create and format heat map subplots"""

        # Define function for plotting one heat map subplot
        def plot_heat_map_subplot(hm_nums_matrix, annot_matrix, ax):
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
        plot_heat_map_subplot(hm_nums_matrix1, annot_matrix1, ax1)
        plot_heat_map_subplot(hm_nums_matrix2, annot_matrix2, ax2)
        plot_heat_map_subplot(hm_nums_matrix3, annot_matrix3, ax3)
        plt.show()


def get_cluster_labels(cc, case, n_clusters, find_opt=False):  # ✓
    """Find clusters of data points in the input tr(S²) and tr(R²) parameter space that
    have smooth mappings with output g_n coefficient"""

    cc.min_max_data()
    if case == "BUMP_h38" or case == "IMPJ_20000":
        cc.rand_sample(n_samples=20000)
    cc.change_dtype()
    cluster_data = cc.pre_cluster_format()
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


def cluster(S_sq_trace, R_sq_trace, g_coeffs, n_clusters, case, g_coeff_name):  # ✓
    """Perform clustering to show smooth mappings"""

    start = timeit.default_timer()
    cc = ClusterClass(S_sq_trace, R_sq_trace, g_coeffs, case, g_coeff_name)
    # if case == "BUMP_h38":
    #     cc.clip_bump_data()
    if case == "IMPJ_20000":
        cc.clip_jet_data()
    min_clusters, max_clusters, sil_avg_list, cluster_labels = \
        get_cluster_labels(cc, case, n_clusters)
    stop = timeit.default_timer()
    print("Clustering run time = ", str(stop - start))
    return cc, sil_avg_list, min_clusters, max_clusters, cluster_labels


def calc_heat_map_nums(cc, cluster_labels, case, nearest_points=True):  # ✓
    """Segment input tr(S²) and tr(R²) parameter space into a rectangular grid and
    find total NUMs in each grid cell"""

    gc = GridClass(case)
    nx, ny, cells_dict = gc.create_cells_dict(cc.S_sq_trace, cc.R_sq_trace)
    hm_nums_list = []
    for i in cells_dict:
        # prox_idx_dict = gc.create_prox_idx_dict(cells_dict, i, cc.S_sq_trace,
        #                                         cc.R_sq_trace)
        prox_idx_dict = gc.create_circ_prox_idx_dict(cells_dict, i, cc.S_sq_trace,
                                                     cc.R_sq_trace)
        if nearest_points is True:
            prox_idx_dict = gc.create_near_n_idx_dict(cells_dict, i, prox_idx_dict,
                                                      cc.S_sq_trace, cc.R_sq_trace)
        _, total_nums = \
            gc.create_flag_nums_dict(cells_dict, i, prox_idx_dict, cluster_labels)
        hm_nums_list.append(total_nums)
    annot_matrix, hm_nums_matrix = gc.create_hm_nums_matrix(hm_nums_list, nx, ny)
    return gc, annot_matrix, hm_nums_matrix


def plot_single_plots(S_sq_trace, R_sq_trace, g_coeff, n_clusters, g_coeff_name,
                      case):  # ✓
    """Plot smooth mapping cluster plot and NUM heat map for the results of a single g_n
    coefficient"""

    cc, sil_avg_list, min_clusters, max_clusters, cluster_labels = \
        cluster(S_sq_trace, R_sq_trace, g_coeff, n_clusters, case, g_coeff_name)
    cc.plot_sil_avg(sil_avg_list, min_clusters, max_clusters)
    cc.plot_clusters(cluster_labels, n_clusters, g_coeff_name)
    gc, annot_matrix, hm_nums_matrix = calc_heat_map_nums(cc, cluster_labels, case)
    gc.plot_heat_map(annot_matrix, hm_nums_matrix)


def plot_subplots(S_sq_trace, R_sq_trace, g1, g2, g3, case):  # ✓
    """Plot smooth mapping cluster subplots and NUM heat map for the results of all
    three g_n coefficients"""

    # Cluster all three g coefficients ✓
    n_clusters1, n_clusters2, n_clusters3 = 3, 3, 3
    if case == "PHLL_case_1p5":
        assert n_clusters1 == n_clusters2 == n_clusters3 == 3
    elif case == "BUMP_h38":
        assert n_clusters1 == n_clusters2 == n_clusters3 == 4
    elif case == "IMPJ_20000":
        assert n_clusters1 == n_clusters2 == n_clusters3 == 3
    else:
        Exception("Silhouette score analysis not undertaken for this case")

    cc1, sil_avg_list1, min_clusters, max_clusters, cluster_labels1 = \
        cluster(S_sq_trace, R_sq_trace, g1, n_clusters1, case, "g1")
    cc2, sil_avg_list2, min_clusters, max_clusters, cluster_labels2 = \
        cluster(S_sq_trace, R_sq_trace, g2, n_clusters2, case, "g2")
    cc3, sil_avg_list3, min_clusters, max_clusters, cluster_labels3 = \
        cluster(S_sq_trace, R_sq_trace, g3, n_clusters3, case, "g3")

    # Check S_sq_trace and R_sq_trace are the same across all clusters ✓
    assert np.array_equal(cc1.S_sq_trace, cc2.S_sq_trace) is True
    assert np.array_equal(cc2.S_sq_trace, cc3.S_sq_trace) is True
    assert np.array_equal(cc1.R_sq_trace, cc2.R_sq_trace) is True
    assert np.array_equal(cc2.R_sq_trace, cc3.R_sq_trace) is True

    # Plot average silhouette score for all g coefficients on same plot ✓
    # cc1.plot_all_sil_avg(sil_avg_list1, sil_avg_list2, sil_avg_list3, min_clusters,
    #                      max_clusters)

    # Plot cluster subplots ✓
    # cc1.plot_clusters_subplots(cluster_labels1, cluster_labels2, cluster_labels3,
    #                            n_clusters1, n_clusters2, n_clusters3)

    # Plot clusters in separate subplots ✓
    # cc1.plot_clusters_sep(cluster_labels1, cluster_labels2, cluster_labels3,
    #                       n_clusters1, n_clusters2, n_clusters3, cc1.g_coeffs,
    #                       cc2.g_coeffs, cc3.g_coeffs, case)

    # Calculate NUM instances for heat map plotting all three g coefficients ✓
    gc1, annot_matrix1, hm_nums_matrix1 = calc_heat_map_nums(cc1, cluster_labels1, case)
    gc2, annot_matrix2, hm_nums_matrix2 = calc_heat_map_nums(cc2, cluster_labels2, case)
    gc3, annot_matrix3, hm_nums_matrix3 = calc_heat_map_nums(cc3, cluster_labels3, case)

    # Plot heat map subplots ✓
    gc1.plot_heat_map_subplots(annot_matrix1, hm_nums_matrix1, annot_matrix2,
                               hm_nums_matrix2, annot_matrix3, hm_nums_matrix3)

    # Save heat map matrices
    np.save('IMPJ_20000_g1_hm_matrix_two_inputs', annot_matrix1)
    np.save('IMPJ_20000_g2_hm_matrix_two_inputs', annot_matrix2)
    np.save('IMPJ_20000_g3_hm_matrix_two_inputs', annot_matrix3)

    print("finish")
