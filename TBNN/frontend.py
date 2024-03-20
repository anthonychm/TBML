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
              zonal_test_dataset=np.nan, version="v1"):
    # Define parameters
    num_hid_layers = 5  # Number of hidden layers in the TBNN, default = 3
    num_hid_nodes = [20] * num_hid_layers  # Number of nodes in the hidden layers given
    # as a vector, default = [5, 5, 5]
    num_tensor_basis = 3  # Number of tensor bases for bij prediction, also the num of
    # output nodes. For 2D = 3, for 3D = 10
    max_epochs = 100000  # Max number of training epochs, default = 2000
    min_epochs = 100  # Min number of training epochs, default = 1000
    interval = 4  # Frequency of epochs at which the model is evaluated on validation
    # dataset, default = 100
    avg_interval = 3  # Number of intervals averaged over for early stopping criteria,
    # default = 4
    enforce_realiz = True  # Boolean for enforcing realizability on Reynolds stresses,
    # default = True
    num_realiz_its = 5  # Number of iterations for realizability enforcing, default = 5

    # Define advanced parameters
    af = ["SiLU"] * num_hid_layers  # Nonlinear activation function, default =
    # ["ELU"]*(num_hid_layers)
    af_params = ["inplace=False"] * num_hid_layers  # Parameters of the nonlinear
    # activation function, default = ["alpha=1.0, inplace=False"]*(num_hid_layers)
    weight_init = "kaiming_normal_"  # Weight initialiser algorithm,
    # default = "kaiming_normal_"
    weight_init_params = "nonlinearity=leaky_relu"  # Arguments of the weight initialiser
    # algorithm, default = "nonlinearity=leaky_relu"
    init_lr = 0.01  # Initial learning rate, default = 0.01
    lr_scheduler = "ExponentialLR"  # Learning rate scheduler, default = ExponentialLR
    lr_scheduler_params = "gamma=0.95"  # Parameters of learning rate scheduler,
    # default = "gamma=0.9"
    loss = "MSELoss"  # Loss function, default = "MSELoss"
    optimizer = "Adam"  # Optimizer algorithm, default = "Adam"
    batch_size = 32  # Training batch size, default = 16

    # Define TBNN inputs
    two_invars = True  # Only include the first two invariants tr(S²) and tr(R²)
    incl_p_invars = False  # Include pressure invariants in inputs
    incl_tke_invars = False  # Include tke invariants in inputs
    incl_input_markers = False  # Include scalar markers in inputs
    num_input_markers = None  # Number of scalar markers in inputs
    rho = 1.514  # Density of air at -40C with nu = 1e-5 m²/s

    # Define splitting of training, validation and testing datasets
    train_test_rand_split = False  # Randomly split entire database for training and
    # testing, default = False
    train_test_split_frac = None  # Fraction of entire database used for training and
    # validation if train_test_rand_split = True, default = 0
    train_valid_rand_split = False  # Randomly split training database for training and
    # validation, default = False
    train_valid_split_frac = None  # Fraction of training database used for actual
    # training while the other fraction is used for validation if
    # train_valid_rand_split = True, default = 0.

    # Set case lists if train_test_rand_split and train_valid_rand_split = False
    train_list = [1, 3]
    valid_list = [2.5]
    test_list = [2, 4]

    # Other
    num_dims = 2  # Number of dimensions in dataset
    num_seeds = 1  # Number of reproducible TBNN predictions to save
    print_freq = 4  # Console print frequency

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    folder_path = create_parent_folders()
    start = timeit.default_timer()

    if version == "v1":
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

    elif version == "v2":
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
    database_name = "FBFS5_full_set_no_walls.txt"  # Data source
    db2case = case_dicts.case_dict_names()
    case_dict, _, num_skip_rows = db2case[database_name]  # ✓
    database = np.loadtxt(database_name, skiprows=num_skip_rows)

    # Run TBNN ✓
    tbnn_main(database, case_dict)  # ✓




