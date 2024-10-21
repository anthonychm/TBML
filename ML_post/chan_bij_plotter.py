"""
This script plots line profiles of bij predicted by ML vs. RANS SST vs. LES in channel
flow cases.
"""

import matplotlib.pyplot as plt
import chan_plotter_core as cpc


def main():
    # Specify ML results identifiers e.g. [first profile, second profile, etc]
    trial = [46]  # [27, 46]
    seed = [1]  # [1, 1]
    num_kernels = [2]  # [None, 2]
    model = ["TBmix"]  # ["TBNN", "TBmix"]
    Re = 395

    # Specify x and y axes
    dim_var_name = "y+"
    bij_comp = "b12"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # Load RANS and LES results for comparison
    cmp_loader = cpc.ResultsLoader(locals())
    rans_result, true_result = cmp_loader.load_cmp_results()
    rans_Cy = cmp_loader.extract_chan_coords(rans_result)
    true_Cy = cmp_loader.extract_chan_coords(true_result)
    rans_bij, true_bij = cmp_loader.extract_cmp_bij(rans_result, true_result)

    # Preprocess RANS plotting data
    rans_plotter = cpc.ChanLinePlotter(locals(), rans_Cy, bij=rans_bij)
    rans_plotter.calc_yplus()
    rans_plotter.halve_chan(y_var=rans_plotter.y_var, bij=rans_plotter.bij)

    # Preprocess LES plotting data
    true_plotter = cpc.ChanLinePlotter(locals(), true_Cy, bij=true_bij)
    true_plotter.calc_yplus()
    true_plotter.halve_chan(y_var=true_plotter.y_var, bij=true_plotter.bij)

    # Initialise plotting
    plt.figure(figsize=(3.5, 1.75))
    bij_comp_dict = {"b11": 0, "b12": 1, "b13": 2, "b22": 4, "b23": 5, "b33": 8}
    col_idx = bij_comp_dict[bij_comp]
    # true_plotter.fill_oti_subdomain(plt, ymin, ymax)

    # Plot RANS and LES bij results
    plt.plot(rans_plotter.y_var, rans_plotter.bij[:, col_idx], 'r',
             linewidth=1, label="RANS")
    plt.plot(true_plotter.y_var, true_plotter.bij[:, col_idx], 'g', linewidth=1,
             label="LES")

    # Recursively load and plot ML results
    for ml_count in range(len(trial)):
        ml_loader = cpc.ResultsLoader(locals())
        ml_loader.get_identifiers(ml_count)
        ml_loader.declare_path()

        # Extract coordinates and predictions of bij from ML
        if ml_loader.model == "TBNN":
            ml_result = ml_loader.load_tbnn_results()
            ml_Cy, ml_bij = ml_result[:, 0], ml_result[:, 1:]
        elif ml_loader.model == "TBmix":
            mu_bij_all, sigma = ml_loader.load_tbmix_results()
            ml_Cy, ml_bij = sigma[:, 0], None  # ml_bij set to None as mu_bij_all is used
        else:
            raise Exception("Invalid model name")

        # Preprocess ML plotting data
        if ml_loader.model == "TBNN":
            ml_plotter = cpc.ChanLinePlotter(locals(), ml_Cy, bij=ml_bij)
            ml_plotter.calc_yplus()
            ml_plotter.halve_chan(y_var=ml_plotter.y_var, bij=ml_plotter.bij)
        elif ml_loader.model == "TBmix":
            ml_plotter = cpc.ChanLinePlotter(locals(), ml_Cy, bij=ml_bij, mix_bij=True)
            ml_plotter.calc_yplus()
            ml_plotter.halve_chan(y_var=ml_plotter.y_var,
                                  mu_bij_all=ml_plotter.mu_bij_all,
                                  sigma=ml_plotter.sigma)
        else:
            raise Exception("Invalid model name")

        # Plot bij results from ML
        if ml_loader.model == "TBNN":
            plt.plot(ml_plotter.y_var, ml_plotter.bij[:, col_idx], 'k', linewidth=1,
                     label=ml_loader.model)
        elif ml_loader.model == "TBmix":
            mu = ml_plotter.extract_mu_bij_comp(col_idx)
            ml_plotter.plot_kernel_bij(plt, mu)

    # Format bij plot
    true_plotter.fill_oti_subdomain(plt)
    true_plotter.fmt_bij_plot(plt, dim_var_name, bij_comp)


if __name__ == "__main__":
    main()
