import numpy as np
import sys

sys.path.append("../Driver")
sys.path.append("../TBNN")
from Driver import zonal_data_plotter
from TBNN.calculator import PopeDataProcessor


class ResultsLoader:
    def __init__(self, locals_dict):
        self.trial = locals_dict['trial']
        self.seed = locals_dict['seed']
        self.model = locals_dict['model']
        self.kernel = locals_dict['kernel']
        self.dataset = locals_dict['dataset']
        self.case = locals_dict['case']
        self.plot_var_name = locals_dict['plot_var_name']
        self.num_coords = locals_dict['num_coords']
        self.path = None

    def declare_ml_results_path(self):
        self.path = "../" + self.model + "/Saved results/MDN Feb 2024 " + \
                    self.dataset[:4] + " trials/Trial " + str(self.trial) + "/Trial" + \
                    str(self.trial) + "_seed" + str(self.seed) + "_"

    def load_tbnn_results(self):
        # Load tbnn results - these are deterministic
        assert self.model == "TBNN"
        return np.loadtxt(self.path + "TBNN_test_prediction_bij.txt", skiprows=1)

    def load_tbmix_mix_coeff_results(self):
        assert self.model == "TBmix"
        assert "mix_coeff" in self.plot_var_name
        return np.loadtxt(self.path + "mixing_coefficients.txt", skiprows=1)

    def load_tbmix_mean_bij_results(self):
        assert self.model == "TBmix"
        return np.loadtxt(self.path + "mu_bij_from_kernel_" + str(self.kernel) +
                          ".txt", skiprows=1)

    def load_tbmix_sigma_results(self):
        assert self.model == "TBmix"
        assert "sigma" in self.plot_var_name
        return np.loadtxt(self.path + "sigma.txt", skiprows=1)

    def load_true_results(self):
        return np.loadtxt("../Datasets/" + self.dataset + "_dataset.txt", skiprows=1)

    def load_rans_results(self):
        return np.loadtxt("../Datasets/RANS_ref_tables/" + self.dataset +
                          "_rans_ref_table.txt", skiprows=1)

    def calc_start_end(self):
        # Extract start and end rows
        if self.dataset == "FBFS5":
            case_list, num_rows = [1800, 3600, 4500, 5400, 7200], 80296
        elif self.dataset == "PHLL4":
            case_list, num_rows = ["_0p5", "_1p0", "_1p2", "_1p5"], 14751
        elif self.dataset == "MVEN6":
            case_list = ["_0p5", "_1p0", "_1p2", "_1p5", "3500", "3600"]
            num_rows = [14751, 14751, 14751, 14751, 9216, 80296]
            start_row = 0
            for i in range(len(case_list)):
                end_row = start_row + num_rows[i]
                if self.case[-4:] == case_list[i]:
                    return start_row, end_row
                else:
                    start_row = end_row
            raise Exception("Case not found in case_list")
        else:
            raise Exception("Dataset not supported in this method")
        idx = case_list.index(self.case[-4:])
        return idx * num_rows, (idx + 1) * num_rows

    def extract_coords(self, result):
        # Extract Cx and Cy coordinates
        start_row, end_row = self.calc_start_end()
        return result[start_row:end_row, :self.num_coords]

    def extract_rans_bij(self, rans_result):
        start_row, end_row = self.calc_start_end()
        return rans_result[start_row:end_row, self.num_coords:self.num_coords+9]

    def extract_true_bij(self, true_result):
        true_tauij = true_result[:, -9:]
        true_bij = PopeDataProcessor.calc_true_output(true_tauij, output_var="bij")
        start_row, end_row = self.calc_start_end()
        return true_bij[start_row:end_row, :]


class ContourPlotter:
    def __init__(self, case, plot_var_name, coords):
        self.case = case
        self.plot_var_name = plot_var_name
        self.coords = coords
        self.db_dict = None

    def create_non_zonal_database(self):
        # Create non-zonal database dictionary
        db = np.full((len(self.coords), 23), np.nan)  # fill only first two columns
        db[:, 1:3] = self.coords
        self.db_dict = {"non_zonal": db}

    def extract_bij_comp_idx(self):
        bij_comp_dict = {"b11": 0, "b12": 1, "b13": 2, "b22": 4, "b23": 5, "b33": 8}
        return bij_comp_dict[self.plot_var_name]

    def plot_contour(self, plot_var):
        # Create grids for contour plotting
        zdp = zonal_data_plotter.ZonalDataPlotter([self.case], "non_zonal", self.db_dict)
        if "PHLL" in self.case:
            x_grid, y_grid, z_grid = \
                zdp.create_sorted_contourf_grids("non_zonal", plot_var, self.case)
        elif "FBFS" or "DUCT" in self.case:
            x_grid, y_grid, z_grid = \
                zdp.create_allocated_contourf_grids("non_zonal", plot_var, self.case)
        else:
            raise Exception("No valid function for grid creation")

        # Plot contour
        zdp.plot_grid_contourf(x_grid, y_grid, z_grid, self.plot_var_name)
