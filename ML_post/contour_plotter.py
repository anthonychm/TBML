"""
This script is used to plot:
1. ML prediction results from tbnn/tbmix
2. RANS baseline bij results
3. "True" bij results from LES/DNS/experiments
"""

import contour_plotter_core as cpc

# Declare variables
trial: None or int = 66
seed: None or int = 1
model: str = "TBmix"  # RANS, true, TBNN, TBmix
kernel: None or int = 1
dataset: str = "PHLL4"
case: str = "PHLL_case_1p2"
plot_var_name: str = "sigma"  # bij, mix_coeff or sigma
num_coords: int = 3

# Initialise results loader and load results
loader = cpc.ResultsLoader(locals())
if model == "RANS":
    # Load RANS coordinates and bij array
    rans_result = loader.load_rans_results()
    coords = loader.extract_coords(rans_result)
    arr = loader.extract_rans_bij(rans_result)

elif model == "true":
    # Load true coordinates and bij array
    true_result = loader.load_true_results()
    coords = loader.extract_coords(true_result)
    arr = loader.extract_true_bij(true_result)

elif model == "TBNN":
    # Load TBNN coordinates and bij array
    loader.declare_ml_results_path()
    tbnn_result = loader.load_tbnn_results()
    coords, arr = tbnn_result[:, :num_coords], tbnn_result[:, num_coords:]

elif model == "TBmix":
    # Load TBmix coordinates and array of variable for plotting
    loader.declare_ml_results_path()
    if "mix_coeff" in plot_var_name:
        tbmix_result = loader.load_tbmix_mix_coeff_results()
    elif plot_var_name[0] == "b":
        tbmix_result = loader.load_tbmix_mean_bij_results()
    elif plot_var_name == "sigma":
        tbmix_result = loader.load_tbmix_sigma_results()
    else:
        raise Exception("Invalid plot variable name")
    coords, arr = tbmix_result[:, :num_coords], tbmix_result[:, num_coords:]

else:
    raise Exception("Invalid model name")

# Initialise contour plotter and plot contour
if "PHLL" in case:
    coords = coords[:, :2]
elif "FBFS" in case:
    coords = coords[:, :2]
    coords[:, 0] = (coords[:, 0] + 0.063) / 0.018
    coords[:, 1] = (coords[:, 1] + 0.003) / 0.018
elif "DUCT" in case:
    coords = coords[:, -2:]
plotter = cpc.ContourPlotter(case, plot_var_name, coords)
plotter.create_non_zonal_database()
if plot_var_name[0] == "b":
    col_idx = plotter.extract_bij_comp_idx()
    plotter.plot_contour(arr[:, col_idx])
elif "mix_coeff" in plot_var_name or "sigma" in plot_var_name:
    plotter.plot_contour(arr[:, kernel-1])
else:
    raise Exception("Invalid plot variable name")
