"""
===============================================
===== This code was written by Anthony Man ====
====== The University of Manchester, 2022 =====
===============================================

This program performs all the steps for training and testing a Tensor Basis Neural Network
(TBNN) for any kind of turbulent flow problem.

Input 1: Invariants of Sij, Rij, gradp and gradk.
Input 2: Tensor basis of Sij and Rij.
Sij = Mean strain rate tensor, Rij = Mean rotation rate tensor (both Sij and Rij are
non-dimensionalised by k/eps). gradp = pressure gradient, gradk = TKE gradient.
Sij, Rij, gradp and gradk are from RANS simulations.

Output: Reynolds stress anisotropy tensor bij.
True bij can be from highly-resolved LES or DNS or experimental results. The TBNN aims to
produce this true output.

Reference for data driven turbulence modelling TBNN implementation:
Ling, J., Kurzawski, A. and Templeton, J., 2016. Reynolds averaged turbulence modelling
using deep neural networks with embedded invariance. Journal of Fluid Mechanics, 807,
pp.155-166.

Syntax and run checked by AM, 17/12/2022
PoF Oct 2022 models run using:
Local: Python 3.10.9, numpy 1.23.5, torch 1.13.0 and pandas 1.5.2
CSF4: Python 3.10.4, numpy 1.22.3, torch 1.13.1 and pandas 1.4.2
"""

import numpy as np
import timeit
import case_dicts
from pred_iterator import preprocessing, trial_iter, trial_iter_v2
from results_writer import write_time, create_parent_folders


def tbnn_main(database, case_dict, incl_zonal_markers=False, num_zonal_markers=0,
              zones=np.nan, zonal_train_dataset=np.nan, zonal_valid_dataset=np.nan,
              zonal_test_dataset=np.nan, version="non_zonal"):
    # Define parameters
    num_hid_layers = 3  # Num. of hidden layers, d = 3
    num_hid_nodes = [10] * num_hid_layers  # Num. of hidden nodes, d = [5, 5, 5]
    max_epochs = 2000  # Max num. of epochs, d = 2000
    min_epochs = 15  # Min num. of epochs, d = 1000
    interval = 2  # Model undertakes validation after every interval of epochs, d = 100
    avg_interval = 3  # Num. of intervals averaged over for early stopping, d = 4
    enforce_realiz = True  # Enforce realizability on Reynolds stresses, d = True
    num_realiz_its = 5  # Num. of iterations for realizability enforcing, d = 5

    # Define advanced parameters
    af = ["ReLU"] * num_hid_layers  # Activation functions, d = ["ELU"]*(num_hid_layers)
    af_params = None  # Activation function parameters, d = None
    weight_init = "kaiming_normal_"  # Weight initialiser, d = "kaiming_normal_"
    weight_init_params = "nonlinearity=leaky_relu"  # Weight initialiser params, d = "nonlinearity=leaky_relu"
    init_lr = 0.001  # Initial learning rate, d = 0.01
    lr_scheduler = "ExponentialLR"  # Learning rate scheduler, d = "ExponentialLR"
    lr_scheduler_params = "gamma=1"  # Learning rate scheduler params, d = "gamma=0.9"
    loss = "MSELoss"  # Loss function, d = "MSELoss"
    optimizer = "Adam"  # Optimizer, d = "Adam"
    batch_size = 32  # Batch size, d = 16

    # Define TBNN inputs
    two_invars = True  # Only include the first two invariants tr(S²) and tr(R²)
    incl_p_invars = False  # Include pressure invariants in inputs
    incl_tke_invars = False  # Include tke invariants in inputs
    incl_input_markers = False  # Include scalar markers in inputs
    num_input_markers = None  # Number of scalar markers in inputs
    rho = 1.514  # Density of air at -40C with nu = 1e-5 m²/s
    num_tensor_basis = 3  # Num. of tensor bases; for 2D flow = 3, for 3D flow = 10

    # Define splitting of training, validation and testing datasets
    train_test_rand_split = False  # Randomly split entire database for training and
    # testing, d = False
    train_test_split_frac = None  # Fraction of entire database used for training and
    # validation if train_test_rand_split = True, d = 0
    train_valid_rand_split = False  # Randomly split training database for training and
    # validation, d = False
    train_valid_split_frac = None  # Fraction of training database used for actual
    # training while the other fraction is used for validation if
    # train_valid_rand_split = True, d = 0.

    # Set case lists if train_test_rand_split and train_valid_rand_split = False
    train_list = [1.5]
    valid_list = [1]
    test_list = [1.2]

    # Other
    num_dims = 2  # Num. of dimensions in dataset
    num_seeds = 1  # Num. of reproducible TBNN predictions to save
    print_freq = 1  # Print frequency to console

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    folder_path = create_parent_folders()
    start = timeit.default_timer()

    if version == "non_zonal":
        coords, x, tb, y, num_inputs = \
            preprocessing(database, num_dims, num_input_markers, num_zonal_markers,
                          two_invars, incl_p_invars, incl_tke_invars, incl_input_markers,
                          incl_zonal_markers, rho, num_tensor_basis, enforce_realiz,
                          num_realiz_its)  # ✓
        user_vars = locals()
        current_folder = \
            trial_iter(num_seeds, coords, x, tb, y, train_list, valid_list, test_list,
                       train_valid_rand_split, train_valid_split_frac,
                       train_test_rand_split, train_test_split_frac, num_tensor_basis,
                       num_hid_layers, num_hid_nodes, af, af_params, init_lr, lr_scheduler,
                       lr_scheduler_params, weight_init, weight_init_params, max_epochs,
                       min_epochs, interval, avg_interval, loss, optimizer, batch_size,
                       enforce_realiz, num_realiz_its, folder_path, user_vars, print_freq,
                       case_dict, num_inputs)  # ✓

    elif version == "zonal":
        for zone in zones:
            coords_train, x_train, tb_train, y_train, num_inputs = \
                preprocessing(zonal_train_dataset[zone][:, 1:], num_dims, num_input_markers,
                              num_zonal_markers, two_invars, incl_p_invars, incl_tke_invars,
                              incl_input_markers, incl_zonal_markers, rho,
                              num_tensor_basis, enforce_realiz, num_realiz_its)  #

            coords_valid, x_valid, tb_valid, y_valid, num_inputs = \
                preprocessing(zonal_valid_dataset[zone][:, 1:], num_dims, num_input_markers,
                              num_zonal_markers, two_invars, incl_p_invars, incl_tke_invars,
                              incl_input_markers, incl_zonal_markers, rho,
                              num_tensor_basis, enforce_realiz, num_realiz_its)  #

            coords_test, x_test, tb_test, y_test, num_inputs = \
                preprocessing(zonal_test_dataset[zone][:, 1:], num_dims, num_input_markers,
                              num_zonal_markers, two_invars, incl_p_invars, incl_tke_invars,
                              incl_input_markers, incl_zonal_markers, rho,
                              num_tensor_basis, enforce_realiz, num_realiz_its)  #

            user_vars = locals()
            current_folder = \
                trial_iter_v2(num_seeds, x_train, tb_train, y_train, x_valid, tb_valid,
                              y_valid, coords_test, x_test, tb_test, y_test,
                              num_tensor_basis, num_hid_layers, num_hid_nodes, af,
                              af_params, init_lr, lr_scheduler, lr_scheduler_params,
                              weight_init, weight_init_params, max_epochs, min_epochs,
                              interval, avg_interval, loss, optimizer, batch_size,
                              enforce_realiz, num_realiz_its, folder_path, user_vars,
                              print_freq, num_inputs)  #
    else:
        raise Exception("Invalid version")
    stop = timeit.default_timer()
    write_time(start, stop, folder_path, current_folder)  # ✓
    print("TBNN finished")


if __name__ == "__main__":

    # Load database and associated dictionary ✓
    database_name = "PHLL4_dataset.txt"  # Data source
    db2case = case_dicts.case_dict_names()
    case_dict, _, num_skip_rows = db2case[database_name]  # ✓
    database = np.loadtxt(database_name, skiprows=num_skip_rows)

    # Run TBNN ✓
    tbnn_main(database, case_dict)  # ✓




