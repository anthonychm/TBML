"""
This script calculates values of zonal markers across the entire flow domain.
Version 1 (for Apr 2023)
"""

import numpy as np
import matplotlib.pyplot as plt


def calc_zonal_markers():
    # Choose CFD case from: "BUMP_h20" "BUMP_h26" "BUMP_h31", "BUMP_h38", "BUMP_h42", "CBFS_13700",
    # "CNDV_12600", "CNDV_20580", "PHLL_case_0p5", "PHLL_case_0p8", "PHLL_case_1p0", "PHLL_case_1p2", "PHLL_case_1p5"
    # "DUCT_1100", "DUCT_1150", "DUCT_1250", "DUCT_1300", "DUCT_1350", "DUCT_1400", "DUCT_1500", "DUCT_1600"
    # "DUCT_1800", "DUCT_2000", "DUCT_2205", "DUCT_2400" "DUCT_2600", "DUCT_2900", "DUCT_3200", "DUCT_3500"
    # "FBFS_1800", "FBFS_3600", "FBFS_4500", "FBFS_5400", "FBFS_7200"
    case = "FBFS_7200"

    parent_path = get_parent_path(case)
    var_dict = load_marker_data_apr2023(case, parent_path, ["S", "R", "k", "Ux", "Uy", "Uz", "wall_dist"])  # ✓
    S, R, k, Ux, Uy, Uz, wall_dist = unpack_var_dict_calc(var_dict)  # ✓
    nu = get_nu(case)
    #nd_Q = calc_nd_Q(S, R)
    #nd_TI = calc_nd_TI(k, Ux, Uy, Uz)
    #nd_Ux = calc_nd_Ux(Ux, Uy, Uz)
    Re_y = calc_Re_y(k, wall_dist, nu)
    write_vars(case, vars(), "Re_y")  # ✓
    return


def get_parent_path(case):
    # Get parent directory path for the given case ✓
    if any(name in case for name in ["FBFS", "IMPJ"]):
        parent_path = "C:/Users/h81475am/Dropbox (The University of " \
                      "Manchester)/PhD_Anthony_Man/ML Database/" + case[:4]
    elif any(name in case for name in ["BUMP", "CBFS", "CNDV", "PHLL", "DUCT"]):
        parent_path = ("C:/Users/h81475am/Dropbox (The University of "
                       "Manchester)/PhD_Anthony_Man/"
                       "AM_PhD_shared_documents/McConkey data")
    elif any(name in case for name in ["SQCY", "TACY"]):
        parent_path = ("C:/Users/h81475am/Dropbox (The University of Manchester)/"
                       "PhD_Anthony_Man/AM_PhD_shared_documents/Geneva data")
    else:
        raise Exception("No matching parent path for this case")
    return parent_path


def create_var_dict(var_list, parent_path, child_path, case):
    # Create dictionary of variables for calculating the zonal markers ✓
    var_dict = {}
    for count, var in enumerate(var_list):
        try:
            var_dict[var] = np.load(parent_path + child_path + case + "_" + var + ".npy")
        except:
            var_dict[var] = np.loadtxt(parent_path + child_path + case + "_" + var + ".txt")
    return var_dict


def load_marker_data(case, parent_path, var_list):
    # Load data for calculating zonal markers ✓
    if any(name in case for name in ["FBFS", "IMPJ"]):
        var_dict = create_var_dict(var_list, parent_path, "/" + case + "/", case)
    elif any(name in case for name in ["BUMP", "CBFS", "CNDV", "PHLL", "DUCT"]):
        turb_model = "komegasst"
        var_dict = create_var_dict(var_list, parent_path, "/" + turb_model + "/" + turb_model + "_", case)
    elif any(name in case for name in ["SQCY", "TACY"]):
        turb_model = "kepsilon"
        var_dict = create_var_dict(var_list, parent_path, "/" + turb_model + "/" + turb_model + "_", case)
    else:
        raise Exception("No matching data for this case")
    return var_dict


def unpack_var_dict_calc(var_dict):
    # Unpack variable dictionary for calculating zonal markers
    return var_dict["S"], var_dict["R"], var_dict["k"], var_dict["Ux"], var_dict["Uy"], var_dict["Uz"], \
        var_dict["wall_dist"]


def get_nu(case):
    # Get kinematic viscosity, nu for a specified case ✓
    nu_dict = {
        "PHLL": 5e-6,  # ✓
        "BUMP": 2.53e-5,  # ✓
        "CNDV_12600": 7.94e-5,  # ✓
        "CNDV_20580": 4.86e-5,  # ✓
        "CBFS": 7.3e-5,  # ✓
        "FBFS": 1e-5,  # ✓
        "IMPJ": 1.5e-5  # ✓
    }

    for key in nu_dict:
        if key in case:
            nu = nu_dict[key]
            return nu


def calc_S_and_R(parent_path, child_path, case):
    # Calculate dimensional mean strain rate S and mean rotation rate R, both with dimensions 1/s ✓

    # Convert grad_u_flat to 3-dimensional format
    grad_u_flat = np.loadtxt(parent_path + child_path + case + "_gradU.txt")
    num_points = grad_u_flat.shape[0]
    grad_u = np.full((num_points, 3, 3), np.nan)
    for i in range(3):
        for j in range(3):
            grad_u[:, i, j] = grad_u_flat[:, (3*i)+j]

    # Calculate S and R
    S = np.full((num_points, 3, 3), np.nan)
    R = np.full((num_points, 3, 3), np.nan)
    for i in range(num_points):
        S[i, :, :] = 0.5 * (grad_u[i, :, :] + np.transpose(grad_u[i, :, :]))
        R[i, :, :] = 0.5 * (grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))

    # Write S and R as npy files
    np.save(str(case) + '_S.npy', S)
    np.save(str(case) + '_R.npy', R)


def calc_nd_Q(S, R):
    # Calculate non-dimensional Q-criterion, nd_Q = (||R||^2 - ||S||^2)/(||R||^2 + ||S||^2) ✓
    nd_Q = []
    for row in range(S.shape[0]):
        norm_S = 0
        norm_R = 0
        for i in range(2):
            for j in range(2):
                norm_S += S[row, i, j] ** 2
                norm_R += R[row, i, j] ** 2
        norm_S = np.sqrt(norm_S)
        norm_R = np.sqrt(norm_R)
        nd_Q_tmp = ((norm_R ** 2) - (norm_S ** 2))/((norm_R ** 2) + (norm_S ** 2))
        nd_Q.append(nd_Q_tmp)
    return nd_Q


def calc_nd_TI(k, Ux, Uy, Uz):
    # Calculate non-dimensional turbulence intensity, nd_TI = k/(0.5UiUi + k) ✓
    nd_TI = k/((0.5*((Ux ** 2)+(Uy ** 2)+(Uz ** 2)))+k)
    return nd_TI


def calc_nd_Ux(Ux, Uy, Uz):
    # Calculate non-dimensional streamwise velocity, nd_Ux = Ux/mag(U) ✓
    nd_Ux = Ux/np.sqrt((Ux ** 2)+(Uy ** 2)+(Uz ** 2))
    return nd_Ux


def calc_Re_y(k, wall_dist, nu):
    # Calculate turbulence Reynolds number, Re_y = sqrt(k)*y/200nu ✓
    Re_y = (np.sqrt(k)*wall_dist)/(200*nu)
    return Re_y


def write_vars(case, vars_dict, *args):
    # Write results as a .txt file ✓
    for var in args:
        np.savetxt(case + "_" + var + ".txt", vars_dict[var])


def load_zonal_markers(case, parent_path, marker_list):
    # Load calculated zonal markers ✓
    if any(name in case for name in ["FBFS", "IMPJ"]):
        marker_dict = create_var_dict(marker_list, parent_path, "/zonal criteria/", case)
    elif any(name in case for name in ["BUMP", "CBFS", "CNDV", "PHLL", "DUCT"]):
        turb_model = "komegasst"
        marker_dict = create_var_dict(marker_list, parent_path, "/" + turb_model + "/zonal criteria/", case)
    else:
        raise Exception("No matching zonal marker results for this case")
    return marker_dict


def plot_zonal_markers(Cx, Cy, marker, marker_dict):
    # Plot zonal marker with tricontourf ✓

    def get_marker_level(marker):
        level_dict = {
            "nd_Q": 0,
            "nd_TI": 0.5,
            "nd_Ux": 0,
            "Re_y": 1
        }
        return level_dict[marker]

    level = get_marker_level(marker)
    contourf = plt.tricontour(Cx, Cy, marker_dict[marker], levels=[level])
    #plt.colorbar()
    plt.show()
    return contourf


def check_zonal_markers():
    # Plot zonal markers to check them against ParaView iso-volumes  ✓
    case = "FBFS_7200"
    marker = "Re_y"  # nd_Q, nd_TI, nd_Ux, Re_y
    parent_path = get_parent_path(case)
    coords_dict = load_marker_data_apr2023(case, parent_path, ["Cx", "Cy", "Cz"])
    Cx = coords_dict["Cx"]
    Cy = coords_dict["Cy"]
    Cz = coords_dict["Cz"]
    # assert (len(set(Cz)) == 1)

    marker_dict = load_zonal_markers_apr2023(case, parent_path, ["nd_Q", "nd_TI", "nd_Ux", "Re_y"])
    plt.scatter(Cx, Cy, c=marker_dict[marker], s=10, cmap='viridis')
    plt.colorbar()
    plt.show()
    # _ = plot_zonal_markers(Cx, Cy, marker, marker_dict)

    # PHLL, BUMP, CNDV, CBFS, FBFS checked


if __name__ == "__main__":
    check_zonal_markers()
    print("finish")