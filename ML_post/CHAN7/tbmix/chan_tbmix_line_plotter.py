# Checked by AM on 07/08/2024

import matplotlib.pyplot as plt
import numpy as np


def load_results(folder_path, trial, seed, num_kernels):  # ✓
    # Load mixture coefficient results
    mix_coeffs = np.loadtxt(folder_path + "Trials/Trial " + str(trial) + "/Trial" +
                            str(trial) + "_seed" + str(seed) +
                            "_mixing_coefficients.txt", skiprows=1)

    # Load mean bij results of each kernel
    mu_bij_all = np.full((num_kernels, len(mix_coeffs), 10), np.nan)
    for k in range(num_kernels):
        mu_bij_kernel = np.loadtxt(folder_path + "Trials/Trial " + str(trial) + "/Trial" +
                                   str(trial) + "_seed" + str(seed) +
                                   "_mu_bij_from_kernel_" + str(k+1) + ".txt", skiprows=1)
        mu_bij_all[k, :, :] = mu_bij_kernel

    # Load sigma results
    sigma = np.loadtxt(folder_path + "Trials/Trial " + str(trial) + "/Trial" +
                       str(trial) + "_seed" + str(seed) + "_sigma.txt", skiprows=1)

    # Load rans and true result for comparison
    rans_result = np.loadtxt("../CHAN7_rans_ref_table.txt", skiprows=1)
    true_result = np.loadtxt(folder_path + "LES_test_truth_bij.txt", skiprows=1)
    return mix_coeffs, mu_bij_all, sigma, true_result, rans_result


class ChanTbmixLinePlotter:
    def __init__(self, mix_coeffs, mu_bij_all, sigma, true_result, rans_result,
                 Re_number):
        self.mix_coeffs, self.sigma = mix_coeffs, sigma
        self.mu_bij_all = mu_bij_all
        self.Cy = self.mix_coeffs[:, 0]
        self.true_result = true_result
        self.rans_result = rans_result
        self.Re_number = Re_number
        self.yplus, self.half_yplus = np.nan, np.nan

    def extract_rans_result(self):
        Re_numbers = [180, 290, 395, 490, 590, 760, 945]
        num_points = 182
        assert len(self.Cy) == num_points
        start_idx = Re_numbers.index(self.Re_number)*num_points
        self.rans_result = self.rans_result[start_idx:(start_idx+num_points), :]

    def calc_yplus(self):
        tau_w_dict = {180: 0.02030, 290: 0.05107, 395: 0.09313, 490: 0.14541,
                      590: 0.20732, 760: 0.35267, 945: 0.52228}
        utau, nu = np.sqrt(tau_w_dict[self.Re_number]), 1.568e-05
        self.yplus = utau * self.Cy / nu

    def half_channel(self):
        half_len = round(len(self.yplus) / 2)
        self.half_yplus = self.yplus[0:half_len]
        self.mix_coeffs = self.mix_coeffs[0:half_len, :]
        self.mu_bij_all = self.mu_bij_all[:, 0:half_len, :]
        self.sigma = self.sigma[0:half_len, :]
        self.true_result = self.true_result[0:half_len, :]
        self.rans_result = self.rans_result[0:half_len, :]

    def rm_coords(self):
        self.mix_coeffs = self.mix_coeffs[:, 1:]
        self.sigma = self.sigma[:, 1:]

    def mix_coeff_plot(self, dim_var):
        plt.figure(figsize=(3, 3))
        plt.xlim(min(dim_var), max(dim_var))
        plt.ylim(0, 1)
        plt.plot(dim_var, self.mix_coeffs)
        plt.tick_params(labelsize=8)

        plt.show()

    def mean_pred_plot(self, dim_var, bij_comp, num_kernels):
        bij_comp_dict = {"b11": 0, "b12": 1, "b13": 2, "b22": 4, "b23": 5, "b33": 8}
        col = -9+bij_comp_dict[bij_comp]

        plt.figure(figsize=(3, 3))
        plt.plot(dim_var, self.true_result[:, col], 'k', label="LES")

        self.rans_result = self.rans_result[:, :10]
        plt.plot(dim_var, self.rans_result[:, col], 'r', label="RANS")

        mu = np.full((len(dim_var), num_kernels), np.nan)
        for k in range(num_kernels):
            mu[:, k] = self.mu_bij_all[k, :, col]

        k_count = 0
        for mu_k, sigma_k in zip(mu.T, self.sigma.T):
            k_count += 1
            plt.plot(dim_var, mu_k, 'x', label="kernel " + str(k_count))
            plt.fill_between(dim_var, mu_k+sigma_k, mu_k-sigma_k, alpha=0.4)

        plt.xlim(min(dim_var), max(dim_var))
        plt.ylim(min(self.true_result[:, col]) - 0.1, max(self.true_result[:, col]) + 0.1)
        plt.tick_params(labelsize=8)

        plt.legend(loc='lower right')
        plt.show()


if __name__ == "__main__":
    trial = 37
    seed = 1
    num_kernels = 2
    Re_number = 395
    bij_comp = "b12"
    dim_var = "yplus"

    mix_coeffs, mu_bij_all, sigma, true_result, rans_result = \
        load_results("../../../TBmix/TBNN output data/", trial, seed, num_kernels)  # ✓
    ctlp = ChanTbmixLinePlotter(mix_coeffs, mu_bij_all, sigma, true_result,
                                rans_result, Re_number)  # ✓
    ctlp.extract_rans_result()

    if dim_var == "yplus":  # ✓
        ctlp.calc_yplus()
        ctlp.half_channel()
        ctlp.rm_coords()
        ctlp.mix_coeff_plot(ctlp.half_yplus)
        ctlp.mean_pred_plot(ctlp.half_yplus, bij_comp, num_kernels)
    if dim_var == "y":  # ✓
        ctlp.rm_coords()
        ctlp.mix_coeff_plot(ctlp.Cy)
        ctlp.mean_pred_plot(ctlp.Cy, bij_comp, num_kernels)
