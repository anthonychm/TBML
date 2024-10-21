"""
This script contains classes used in chan_bij_plotter.py and chan_mix_coeff_plotter.py.
"""

import numpy as np
import sys
sys.path.append('../TBNN')
from TBNN.calculator import PopeDataProcessor


class ResultsLoader:
    def __init__(self, locals_dict):
        self.trial = locals_dict['trial']
        self.seed = locals_dict['seed']
        self.Re = locals_dict['Re']
        self.model = locals_dict['model']
        self.bij_comp = locals_dict['bij_comp']
        self.num_kernels = locals_dict['num_kernels']
        self.path = None

    def get_identifiers(self, count):
        # Get identifiers (trial no., seed no., model type) for the current ML result
        self.trial = self.trial[count]
        self.seed = self.seed[count]
        self.model = self.model[count]
        if self.model == "TBmix":
            self.num_kernels = self.num_kernels[count]

    def declare_path(self):
        # Specify folder path of ML model results
        self.path = "../" + self.model + "/Saved results/MDN Feb 2024 CHAN trials/Trial " \
                    + str(self.trial) + "/Trial" + str(self.trial) + "_seed" + \
                    str(self.seed) + "_"

    def load_tbnn_results(self):
        # Load tbnn results - these are deterministic
        assert self.model == "TBNN"
        return np.loadtxt(self.path + "TBNN_test_prediction_bij.txt", skiprows=1)

    def load_tbmix_results(self):
        # Load tbmix component results
        assert self.model == "TBmix"

        # Standard deviation
        sigma = np.loadtxt(self.path + "sigma.txt", skiprows=1)

        # Mean bij from each kernel
        mu_bij_all = np.full((self.num_kernels, len(sigma), 10), np.nan)
        for k in range(self.num_kernels):
            mu_bij_all[k, :, :] = np.loadtxt(self.path + "mu_bij_from_kernel_" +
                                             str(k+1) + ".txt", skiprows=1)

        return mu_bij_all, sigma

    @staticmethod
    def load_cmp_results():
        # Load RANS and LES results for comparison
        rans_result = np.loadtxt("../Datasets/RANS ref tables/CHAN7_rans_ref_table.txt",
                                 skiprows=1)
        true_result = np.loadtxt("../Datasets/CHAN7_dataset.txt", skiprows=1)
        return rans_result, true_result

    def calc_start_end(self):
        # Extract start and end rows corresponding to the Re number
        Re_numbers = [180, 290, 395, 490, 590, 760, 945]
        Re_idx = Re_numbers.index(self.Re)
        num_points = 182
        return Re_idx * num_points, (Re_idx + 1) * num_points

    def extract_chan_coords(self, result):
        # Extract Cy coordinates
        start_row, end_row = self.calc_start_end()
        return result[start_row:end_row, 0]

    def extract_cmp_bij(self, rans_result, true_result):
        # Extract array containing only bij from RANS and LES
        rans_bij = rans_result[:, 1:10]
        true_tauij = true_result[:, -9:]
        true_bij = PopeDataProcessor.calc_true_output(true_tauij, output_var="bij")
        start_row, end_row = self.calc_start_end()
        return rans_bij[start_row:end_row, :], true_bij[start_row:end_row, :]


class ChanLinePlotter:
    def __init__(self, locals_dict, Cy, bij=None, mix_bij=False):
        self.Re = locals_dict['Re']
        self.Re_list = [180, 290, 395, 490, 590, 760, 945]
        self.Cy = Cy
        self.y_var = None
        self.bij = bij
        self.mix_coeffs = None
        self.sigma = None

        if mix_bij is True:
            self.mu_bij_all = locals_dict['mu_bij_all'][:, :, 1:]
            self.sigma = locals_dict['sigma'][:, 1:]
            self.num_kernels = self.mu_bij_all.shape[0]

    def calc_yplus(self):
        # Calculate y+
        tau_w_list = [0.02030, 0.05107, 0.09313, 0.14541, 0.20732, 0.35267, 0.52228]
        tau_w_dict = dict(zip(self.Re_list, tau_w_list))
        utau, nu = np.sqrt(tau_w_dict[self.Re]), 1.568e-05
        self.y_var = utau * self.Cy / nu

    def halve_chan(self, **kwargs):
        # Keep results from the first half of the channel (Cy = 0 to 0.02)
        end_row = round(len(self.y_var) / 2)
        for key, arr in kwargs.items():
            if arr.ndim == 1:
                setattr(self, key, arr[0:end_row])
            elif arr.ndim == 2:
                setattr(self, key, arr[0:end_row, :])
            elif arr.ndim == 3:
                setattr(self, key, arr[:, 0:end_row, :])
            else:
                raise Exception("Number of dimensions not supported")

    def extract_mu_bij_comp(self, col_idx):
        # Extract required component of mean bij
        mu = np.full((len(self.y_var), self.num_kernels), np.nan)
        for k in range(self.num_kernels):
            mu[:, k] = self.mu_bij_all[k, :, col_idx]
        return mu

    def plot_kernel_bij(self, plt, mu):
        # Plot kernel bij results from TBmix
        plt.gca().set_prop_cycle(color=['#d95f02', '#008dff', '#f0c571', '#59a89c',
                                        '#9d2c00'])
        k_count = 0
        for mu_k, sigma_k in zip(mu.T, self.sigma.T):
            k_count += 1
            plt.plot(self.y_var, mu_k, linewidth=1, label="kernel " + str(k_count))
            plt.fill_between(self.y_var, mu_k + sigma_k, mu_k - sigma_k, alpha=0.3)

    def fill_oti_subdomain(self, plt):
        # Fill the y+ range containing one-to-interval relation between alpha and b12
        # where alpha (= (k/eps)*(du/dy)) > 3.4

        # Specify y+ range containing the one-to-interval region
        yplus_oti = [[8.2, 85.2], [8.4, 124.3], [8.0, 163.0], [8.2, 203.7], [7.9, 235.9],
                     [8.0, 307.7], [7.2, 363.0]]
        yplus_oti_dict = dict(zip(self.Re_list, yplus_oti))

        # Plot rectangular fill representing one-to-interval region
        left, right = yplus_oti_dict[self.Re][0], yplus_oti_dict[self.Re][1]
        ymin, ymax = -10, 10
        rect = plt.Rectangle((left, ymin), right-left, ymax-ymin, facecolor="#FF0000",
                             alpha=0.2)
        plt.gca().add_patch(rect)

        # Plot bounding lines around rectangle
        plt.plot([left, left], [ymin, ymax], 'k--')
        plt.plot([right, right], [ymin, ymax], 'k--')

    def fmt_bij_plot(self, plt, xlabel, ylabel):
        # Format bij plot
        plt.xlim(min(self.y_var), max(self.y_var))
        plt.ylim(-0.175, 0)
        plt.xscale('log')
        plt.tick_params(labelsize=6)
        # plt.legend(loc='upper center')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def fmt_mix_coeffs_plot(self, plt, xlabel):
        # Format mixture coefficients plot
        plt.xlim(min(self.y_var), max(self.y_var))
        plt.ylim(0, 1)
        plt.tick_params(labelsize=8)
        plt.legend(loc='lower right')
        plt.xlabel(xlabel)
        plt.ylabel("Prior probability")
        plt.show()
