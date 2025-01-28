"""
This script calculates heuristics as supplementary inputs for data-driven RANS modelling.
"""

import numpy as np
import Datasets.Dataset_creator_scripts.dataset_creator_class as dcc
import Utils.heuristic_calculator as hc


def main(case_list, write=True):
    for case in case_list:
        # Get path dictionary
        gdl = dcc.GeneralDataLoader
        path_dict = gdl.get_path_dict_from_case(case)

        # Get common flow variables
        prefix = path_dict["rans"] + case + "_"
        k = gdl.load_npy_or_txt(prefix, "k")
        eps = gdl.load_npy_or_txt(prefix, "epsilon")
        gradU = gdl.load_npy_or_txt(prefix, "gradU")
        Ux = gdl.load_npy_or_txt(prefix, "Ux")
        Uy = gdl.load_npy_or_txt(prefix, "Uy")
        Uz = gdl.load_npy_or_txt(prefix, "Uz")
        wall_dist = gdl.load_npy_or_txt(prefix, "wall_dist")

        # Get pressure gradients
        gradPx = gdl.load_npy_or_txt(prefix, "gradpx")
        gradPy = gdl.load_npy_or_txt(prefix, "gradpy")
        gradPz = gdl.load_npy_or_txt(prefix, "gradpz")

        # Get turbulent kinetic energy gradients
        gradkx = gdl.load_npy_or_txt(prefix, "gradkx")
        gradky = gdl.load_npy_or_txt(prefix, "gradky")
        gradkz = gdl.load_npy_or_txt(prefix, "gradkz")

        # Calculate intermediate variables if not available
        S, R = hc.VariableCalculator.calc_S_and_R(gradU)
        nu = hc.MiscMethods.get_nu(case)

        # Calculate heuristic variables
        h_calc = hc.HeuristicCalculator
        nd_Q = h_calc.calc_nd_Q(S, R)
        nd_TI = h_calc.calc_nd_ti(k, Ux, Uy, Uz)
        Re_y = h_calc.calc_Re_y(k, wall_dist, nu)
        visc_ratio = h_calc.calc_visc_ratio(k, eps, nu)
        gradp_sl = h_calc.calc_gradp_streamline(Ux, Uy, Uz, gradPx, gradPy, gradPz)
        ts_ratio = h_calc.calc_time_scale_ratio(S, k, eps)
        k_ratio = h_calc.calc_k_ratio(Ux, Uy, Uz, gradkx, gradky, gradkz, k, eps, S)
        tau_ratio = h_calc.calc_Re_stress_ratio(k, eps, S)

        # Write heuristic variables
        if write is True:
            for var in ('nd_Q', 'nd_TI', 'Re_y', 'visc_ratio', 'gradp_sl', 'ts_ratio',
                        'k_ratio', 'tau_ratio'):
                np.savetxt(case + "_" + var + ".txt", locals()[var], delimiter=' ')


if __name__ == "__main__":
    main(["PHLL_case_0p5", "PHLL_case_1p0", "PHLL_case_1p2", "PHLL_case_1p5", "FBFS_3600",
          "DUCT_3500"], write=True)
