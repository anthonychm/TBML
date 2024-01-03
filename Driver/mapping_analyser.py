"""
This code is based on zonal_driver_v2.py and zonal_data_plotter.py for analysing the
non-unique mapping between inputs and outputs in the 2D general effective viscosity
hypothesis (GEVH). The Apr 2023 paper shows non-zonal results from this code demonstrated
on the periodic hill case PHLL_case_1p5 and impinging jet case IMPJ_20000.

This code was written by Anthony Man at The University of Manchester.
"""

import num_grid_class as ngc
import num_3d_grid_class as n3dgc
import numpy as np
from scipy.stats import spearmanr
from zonal_core import create_non_zonal_db, create_zonal_db
from zonal_data_plotter import ZonalDataPlotter


def mapping_analyser(case, zone, db):

    # Calculate inputs and outputs of 2D GEVH
    zdp = ZonalDataPlotter([case], zone, db)
    S11, S12, R12 = zdp.calc_tensor_comps(zone)
    S_sq_trace, R_sq_trace = zdp.calc_traces(S11, S12, R12)
    b11, b22, b33, b12 = zdp.calc_bij_comps(zone)
    g1 = zdp.calc_g1(b11, b22, b12, S11, S12)
    g2 = zdp.calc_g2(b11, b22, b12, S11, S12, R12)
    g3 = zdp.calc_g3(b11, b22, S11, S12)
    # np.savetxt('g1.txt', g1)
    # np.savetxt('g2.txt', g2)
    # np.savetxt('g3.txt', g3)

    # Calculate supplementary input variables
    # norm_s, norm_r = zdp.calc_norm_s_and_norm_r(zone)
    # t_ratio = zdp.calc_timescale_ratio(zone, norm_s)
    # visc_ratio = zdp.calc_visc_ratio(zone, nu=1.5e-05, C_mu=0.09)
    # nondim_qcrit = zdp.calc_nondim_qcrit(norm_s, norm_r)
    # invar = zdp.calc_gradk_gradp_invar(zone, [case])
    # nondim_tau_yy = zdp.calc_nondim_tau_yy(zone, [case])

    # Find correlation between bij and g coefficients
    # rho = spearmanr(b12, g3)

    # Calculate optimal outputs of 2D GEVH
    # opt_g1 = zdp.calc_opt_g1(b11, b22, b12, S11, S12)
    # opt_g2 = zdp.calc_opt_g2(opt_g1, b12, S11, S12, R12)
    # opt_g3 = zdp.calc_opt_g3(opt_g1, opt_g2, b11, S11, S12, R12)
    # opt_b11 = (opt_g1*S11) + (opt_g2*-2*S12*R12) + \
    #           (opt_g3*((S11**2)+(S12**2)-((1/3)*S_sq_trace)))
    # opt_b22 = (opt_g1*-S11) + (opt_g2*2*S12*R12) + \
    #           (opt_g3*((S11**2)+(S12**2)-((1/3)*S_sq_trace)))
    # opt_b33 = opt_g3*(-1/3)*S_sq_trace
    # opt_b12 = (opt_g1*S12) + (opt_g2*2*S11*R12)

    # Create scatter plots
    # zdp.create_bij_scatter(b11, b22, b33, b12, S_sq_trace, R_sq_trace)
    # zdp.create_g_coeffs_scatter(g1, g2, g3, S_sq_trace, R_sq_trace)
    # zdp.create_g_coeffs_3d_scatter(g1, g2, g3, S_sq_trace, R_sq_trace, nondim_tau_yy)

    # Create contour plots
    # zdp.var_scatter(zone, visc_ratio)
    # zdp.var_tricontourf(zone, g1)
    # x_grid, y_grid, z_grid = zdp.create_sorted_contourf_grids(zone, g1, case)
    x_grid, y_grid, z_grid = zdp.create_allocated_contourf_grids(zone, g3, case)
    zdp.plot_grid_default_contourf(x_grid, y_grid, z_grid)

    # Perform non-unique mapping analysis
    # ngc.plot_single_plots(S_sq_trace, R_sq_trace, opt_g2, 2, "g2", case)
    # ngc.plot_subplots(S_sq_trace, R_sq_trace, g1, g2, g3, case)
    # n3dgc.plot_3d_subplots(S_sq_trace, R_sq_trace, visc_ratio, g1, g2, g3, case)

    # Create contour plots with scatter overlay
    # low_idx_list, shear_idx_list = zdp.find_low_or_shear_locs(S_sq_trace, R_sq_trace)
    # zdp.plot_scatter_contourf(zone, g3, "g3", case, low_idx_list, shear_idx_list)


# Non-zonal input-output mapping analysis
case = "FBFS_7200"
zone = "non_zonal"

# Zonal input-output mapping analysis
coords_list = ["Cx", "Cy"]
marker_list = ["Cy"]  # "nd_Q", "nd_TI", "nd_Ux", "Re_y"
zones = marker_list + ["fstrm"]

if zone == "non_zonal":
    non_zonal_db, _, _ = create_non_zonal_db([case])
    mapping_analyser(case, "non_zonal", non_zonal_db)
else:
    zonal_db, _, _ = create_zonal_db([case], coords_list, marker_list, zones, num_cols=22)
    for zone in zones:
        mapping_analyser(case, zone, zonal_db)

print("Analysis completed")
