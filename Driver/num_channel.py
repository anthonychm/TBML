"""
This code produces plots for investigating non-unique mapping in tensor-basis machine
learning models applied to 1D channel flow at Re_tau = 945.
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sb
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import num_grid_class as ngc


class NumChan945Analyser:
    def __init__(self, tau_w, nu):
        self.tau_w, self.nu = tau_w, nu
        self.lotw_data = np.loadtxt('lotw_chan945_rans_data.txt', skiprows=1,
                                    delimiter=',')
        self.fsnw_data = np.loadtxt('CHAN7_full_set_no_walls.txt', skiprows=1)[-182:, :]
        self.utau = np.sqrt(tau_w)

    def plot_lotw(self):
        # Prepare data for plotting
        ux = self.lotw_data[:, -1]
        y = self.lotw_data[:, 0]
        uplus = ux/self.utau
        yplus = self.utau * y / self.nu

        # y+ vs u+ spline data
        half_len = round(len(yplus) / 2)
        half_yplus = yplus[0:half_len]
        half_uplus = uplus[0:half_len]
        interp1 = interp1d(half_yplus, half_uplus, kind=2)
        half_yplus_ = np.linspace(min(half_yplus), max(half_yplus), num=6000)
        half_uplus_ = interp1(half_yplus_)

        # Log law data
        kappa, Cplus = 0.41, 5.5
        half_yplus_ll = np.append([0.1], half_yplus[1:])
        half_uplus_ll = (1/kappa)*np.log(half_yplus_ll) + Cplus

        # y+ vs du+/dy+ spline data
        half_duPlusOverdyPlus = np.gradient(half_uplus, half_yplus)
        interp2 = interp1d(half_yplus, half_duPlusOverdyPlus, kind=2)
        half_duPlusOverdyPlus_ = interp2(half_yplus_)

        # Plot curves
        fig, ax1 = plt.subplots()
        ax1.plot(half_yplus_, half_uplus_, linewidth=5)
        ax1.plot(half_yplus_, half_yplus_)
        ax1.plot(half_yplus_ll, half_uplus_ll, 'tab:orange')
        ax2 = ax1.twinx()
        ax2.plot(half_yplus_, half_duPlusOverdyPlus_, 'tab:green')

        ax1.set_xscale('log')
        ax1.set_xlim([0.1, 1000])
        ax1.set_ylim([0, 25])
        ax2.set_xlim([0.1, 1000])
        ax2.set_ylim([0, 1.005])
        fig.show()

        print('finish')

    def calc_alpha(self):
        y = self.lotw_data[1:-1, 0]
        yplus = self.utau * y / self.nu
        half_len = round(len(yplus) / 2)
        half_yplus = yplus[0:half_len]

        k = self.fsnw_data[0:half_len, 0]
        eps = self.fsnw_data[0:half_len, 1]
        dudy = self.fsnw_data[0:half_len, 3]
        alpha = (k / eps) * dudy

        return half_len, half_yplus, alpha

    def calc_true_gn(self, half_len, alpha):
        # Prepare data for plotting
        b11 = self.fsnw_data[0:half_len, -9]
        b22 = self.fsnw_data[0:half_len, -5]
        b12 = self.fsnw_data[0:half_len, -8]
        S12 = alpha/2
        R12 = alpha/2

        # Calculate true gn components
        true_g1 = b12 / S12
        true_g2 = (b22 - b11) / (4 * R12 * S12)
        true_g3 = 3 * (b11 + b22) / (2 * (S12 ** 2))

        return true_g1, true_g2, true_g3

    def plot_tbnn_in_out(self, half_yplus, alpha, true_g1, true_g2, true_g3):
        # Plot curves
        fig, ax = plt.subplots()
        fig.subplots_adjust(right=0.5)
        plt.show()

        interp = interp1d(half_yplus, alpha, kind=2)
        half_yplus_ = np.linspace(min(half_yplus), max(half_yplus), num=6000)
        alpha_ = interp(half_yplus_)
        p1, = ax.plot(half_yplus_, alpha_, "k", label="α")

        twin1, twin2, twin3 = ax.twinx(), ax.twinx(), ax.twinx()
        twin2.spines.right.set_position(("axes", 1.4))
        twin3.spines.right.set_position(("axes", 1.8))
        p2, = twin1.plot(half_yplus, true_g1, "r-.", label="true g₁")
        p3, = twin2.plot(half_yplus, true_g2, "g--", label="true g₂")
        p4, = twin3.plot(half_yplus, true_g3, "b:", label="true g₃")

        ax.set_xscale('log')
        ax.set_xlim(1, 1001)
        ax.set_ylim(0, 4)
        twin1.set_ylim(-0.3, 0)
        twin2.set_ylim(-5, 1)
        twin3.set_ylim(0, 100)

        # ax.set_xlabel("y⁺")
        # ax.set_ylabel("α")
        twin1.set_ylabel("true g₁")
        twin2.set_ylabel("true g₂")
        twin3.set_ylabel("true g₃")

        ax.yaxis.label.set_color(p1.get_color())
        twin1.yaxis.label.set_color(p2.get_color())
        twin2.yaxis.label.set_color(p3.get_color())
        twin3.yaxis.label.set_color(p4.get_color())

        tkw = dict(size=4, width=1.5)
        ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
        twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        twin3.tick_params(axis='y', colors=p4.get_color(), **tkw)
        # ax.tick_params(axis='x', **tkw)

        # ax.legend(handles=[p1, p2, p3, p4])
        plt.show()
        print("finish")


def create_g_coeffs_scatter(true_g1, true_g2, true_g3, alpha):

    def create_subscatter(ax, g, title, vlims):
        ax.set(xlabel="α", title=title, xlim=[0, 4], xticks=range(0, 5, 1),
               ylim=[-0.01, 0.01])
        ax.tick_params(axis='both', which='major', labelsize=8)
        sc = ax.scatter(alpha, [0] * len(alpha), c=g, s=100, cmap='plasma', alpha=0.5,
                        vmin=vlims[0], vmax=vlims[1])
        cbar = ax.figure.colorbar(sc, ax=ax, location="bottom", aspect=100)
        cbar.ax.tick_params(labelsize=8)

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    create_subscatter(ax1, true_g1, "g₁", [-0.3, 0])
    ax2 = fig.add_subplot(3, 1, 2)
    create_subscatter(ax2, true_g2, "g₂", [-5, 1])
    ax3 = fig.add_subplot(3, 1, 3)
    create_subscatter(ax3, true_g3, "g₃", [0, 100])
    plt.show()


class ClusterClass:
    def __init__(self, alpha, g_coeffs, g_coeff_name):
        self.alpha, self.g_coeffs, self.g_coeff_name = alpha, g_coeffs, g_coeff_name
        self.xticks = range(0, 4, 1)
        self.alpha_norm, self.g_norm = np.nan, np.nan

    def min_max_data(self):
        vdict = {"CHAN_945": [[-0.3, 0], [-5, 1], [0, 100]]}

        def min_max_func(var, factor=1):
            var_norm = (factor*(var - min(var)))/(max(var) - min(var))
            return var_norm

        self.alpha_norm = min_max_func(self.alpha, factor=0.1)
        g_coeff_int = int([*self.g_coeff_name][1])
        g_min = vdict["CHAN_945"][g_coeff_int - 1][0]
        g_max = vdict["CHAN_945"][g_coeff_int - 1][1]
        self.g_norm = min_max_func(np.minimum(np.maximum(self.g_coeffs, g_min), g_max))

    def pre_cluster_format(self):
        cluster_data = [[self.alpha_norm[0], self.g_norm[0]]]
        for i in range(1, len(self.alpha_norm)):
            cluster_data.append([self.alpha_norm[i], self.g_norm[i]])
        cluster_data = np.array(cluster_data)
        return cluster_data

    def plot_clusters_subplots(self, cluster_labels1, cluster_labels2, cluster_labels3,
                               n_clusters1, n_clusters2, n_clusters3):

        def plot_cluster_subplot(cluster_labels, n_clusters, ax, title):
            colors = cm.bwr(cluster_labels.astype(float))
            ax.scatter(self.alpha, [0] * len(alpha), c=colors, s=100, alpha=0.5)
            ax.set(xlabel="α", title=title, xticks=self.xticks, xlim=[0, 4],
                   ylim=[-0.01, 0.01])
            ax.tick_params(axis='both', which='major', labelsize=10)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        plot_cluster_subplot(cluster_labels1, n_clusters1, ax1, "g₁")
        plot_cluster_subplot(cluster_labels2, n_clusters2, ax2, "g₂")
        plot_cluster_subplot(cluster_labels3, n_clusters3, ax3, "g₃")
        plt.show()

    def plot_alpha_vs_clusters_subplots(self, true_g1, true_g2, true_g3, cluster_labels1,
                                        cluster_labels2, cluster_labels3):

        def plot_line_subplots(true_gn, cluster_labels, ax, title, ylim):
            colors = cm.bwr(cluster_labels.astype(float))  # bwr
            ax.plot(self.alpha, true_gn, 'k')
            ax.scatter(self.alpha, true_gn, c=colors, s=20, clip_on=False)
            ax.set(xlabel="α", xticks=self.xticks, xlim=[0, 4], title=title, ylim=ylim)
            ax.tick_params(axis='both', which='major', labelsize=10)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plot_line_subplots(true_g1, cluster_labels1, ax1, "g1", ylim=[-0.3, 0])
        plot_line_subplots(true_g2, cluster_labels2, ax2, "g2", ylim=[-4, 0])
        plot_line_subplots(true_g3, cluster_labels3, ax3, "g3", ylim=[0, 22.5])
        plt.show()


def cluster(alpha, g_coeffs, n_clusters, g_coeff_name, find_opt=False):
    cc = ClusterClass(alpha, g_coeffs, g_coeff_name)
    cc.min_max_data()
    cluster_data = cc.pre_cluster_format()
    if find_opt is True:
        min_clusters, max_clusters = 2, 10
        sil_avg_list = ngc.ClusterClass.find_opt_n_clusters(cluster_data, min_clusters,
                                                            max_clusters)
        cluster_labels = False
    else:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete")
        cluster_labels = clusterer.fit_predict(cluster_data)
        min_clusters, max_clusters, sil_avg_list = False, False, False
    return cc, sil_avg_list, min_clusters, max_clusters, cluster_labels


class GridClass:
    def __init__(self):
        self.cell_len, self.x_bounds = 0.25, [0, 4]

    def create_cells_dict(self, alpha):
        nx = round((self.x_bounds[1] - self.x_bounds[0]) / self.cell_len)
        x_intervals = []
        for i in range(nx+1):
            x_intervals.append(self.x_bounds[0] + (self.cell_len*i))

        cells_dict = {i: [] for i in range(nx)}
        cell_count = -1
        for i in range(nx):
            cell_count += 1
            cell_len_lower = x_intervals[i]
            cell_len_upper = x_intervals[i + 1]

            for idx in range(len(alpha)):
                if cell_len_lower < alpha[idx] < cell_len_upper:
                    cells_dict[cell_count].append(idx)

        return nx, cells_dict

    def create_prox_idx_dict(self, cells_dict, cell_idx, alpha):
        prox_idx_dict = {}
        point_idx_list = cells_dict[cell_idx]
        for idx in point_idx_list:
            # Define proximity square limits
            prox_len_lower = alpha[idx] - (self.cell_len / 0.625)
            prox_len_upper = alpha[idx] + (self.cell_len / 0.625)
            prox_idx_list = []

            for prox_idx in range(len(alpha)):
                if prox_len_lower < alpha[prox_idx] < prox_len_upper and idx != prox_idx:
                    prox_idx_list.append(prox_idx)
            prox_idx_dict[idx] = prox_idx_list

        return prox_idx_dict

    def create_near_n_idx_dict(self, cells_dict, cell_idx, prox_idx_dict, alpha,
                               near_n=2):
        # Get distance to neighbouring points for each idx in the cell
        point_idx_list = cells_dict[cell_idx]
        near_n_idx_dict = {k: [] for k in point_idx_list}
        for idx in point_idx_list:
            assert len(prox_idx_dict[idx]) >= near_n
            dist_list = []
            for adj_idx in prox_idx_dict[idx]:
                dist = abs(alpha[idx] - alpha[adj_idx])
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
    def preprocess_hm_nums_list(hm_nums_list):
        annot_list = hm_nums_list[:]
        for i, val in enumerate(hm_nums_list):
            if val == 0:
                hm_nums_list[i] = np.log10(val+0.1)
            else:
                hm_nums_list[i] = np.log10(val)

        annot_list = np.expand_dims(np.array(annot_list), axis=0)
        hm_nums_list = np.expand_dims(np.array(hm_nums_list), axis=0)
        return annot_list, hm_nums_list

    @staticmethod
    def plot_heat_map_subplots(annot_list1, hm_nums_list1, annot_list2, hm_nums_list2,
                               annot_list3, hm_nums_list3):

        # Define function for plotting one heat map subplot
        def plot_heat_map_subplot(hm_nums_list, annot_list, ax):
            hm = sb.heatmap(hm_nums_list, ax=ax, cmap="rocket_r", annot=annot_list,
                            annot_kws={'rotation': 45, 'fontsize': 6}, linewidths=0.5,
                            cbar_kws={'location': 'bottom', 'ticks': range(-1, 3, 1)},
                            vmin=-1, vmax=2, linecolor='black')

            # hm = sb.heatmap(hm_nums_list, ax=ax, cmap="rocket_r", linewidths=0.5,
            #                 cbar_kws={'location': 'bottom', 'ticks': range(-1, 3, 1)},
            #                 vmin=-1, vmax=2, linecolor='black')

            hm.set(xticklabels=[], yticklabels=[])
            hm.tick_params(bottom=False, left=False)

        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        plot_heat_map_subplot(hm_nums_list1, annot_list1, ax1)
        plot_heat_map_subplot(hm_nums_list2, annot_list2, ax2)
        plot_heat_map_subplot(hm_nums_list3, annot_list3, ax3)
        plt.show()
        print("finish")


def calc_heat_map_nums(cc, cluster_labels, nearest_points=True):
    gc = GridClass()
    nx, cells_dict = gc.create_cells_dict(cc.alpha)
    hm_nums_list = []
    for i in cells_dict:
        prox_idx_dict = gc.create_prox_idx_dict(cells_dict, i, cc.alpha)
        if nearest_points is True:
            prox_idx_dict = gc.create_near_n_idx_dict(cells_dict, i, prox_idx_dict, alpha)
        _, total_nums = ngc.GridClass.create_flag_nums_dict(cells_dict, i, prox_idx_dict,
                                                            cluster_labels)
        hm_nums_list.append(total_nums)
    annot_list, hm_nums_list = gc.preprocess_hm_nums_list(hm_nums_list)
    return gc, annot_list, hm_nums_list


def plot_subplots(alpha, true_g1, true_g2, true_g3):
    n_clusters1, n_clusters2, n_clusters3 = 2, 2, 2
    cc1, sil_avg_list1, min_clusters, max_clusters, cluster_labels1 = \
        cluster(alpha, true_g1, n_clusters1, "g1")
    cc2, sil_avg_list2, min_clusters, max_clusters, cluster_labels2 = \
        cluster(alpha, true_g2, n_clusters2, "g2")
    cc3, sil_avg_list3, min_clusters, max_clusters, cluster_labels3 = \
        cluster(alpha, true_g3, n_clusters3, "g3")

    ngc.ClusterClass.plot_all_sil_avg(sil_avg_list1, sil_avg_list2, sil_avg_list3,
                                      min_clusters, max_clusters)
    cc1.plot_alpha_vs_clusters_subplots(true_g1, true_g2, true_g3, cluster_labels1,
                                        cluster_labels2, cluster_labels3)

    cc1.plot_clusters_subplots(cluster_labels1, cluster_labels2, cluster_labels3,
                               n_clusters1, n_clusters2, n_clusters3)
    gc1, annot_list1, hm_nums_list1 = calc_heat_map_nums(cc1, cluster_labels1)
    gc2, annot_list2, hm_nums_list2 = calc_heat_map_nums(cc2, cluster_labels2)
    gc3, annot_list3, hm_nums_list3 = calc_heat_map_nums(cc3, cluster_labels3)
    gc1.plot_heat_map_subplots(annot_list1, hm_nums_list1, annot_list2, hm_nums_list2,
                               annot_list3, hm_nums_list3)


nca = NumChan945Analyser(0.52228, 1.568e-05)
half_len, half_yplus, alpha = nca.calc_alpha()
true_g1, true_g2, true_g3 = nca.calc_true_gn(half_len, alpha)
# create_g_coeffs_scatter(true_g1, true_g2, true_g3, alpha)
plot_subplots(alpha, true_g1, true_g2, true_g3)
# nca.plot_tbnn_in_out(half_yplus, alpha, true_g1, true_g2, true_g3)
# nca.plot_lotw()
