"""
===============================================
===== This code was written by Anthony Man ====
====== The University of Manchester, 2024 =====
===============================================

This program performs all the steps for training, validating, and testing a tensor basis
mixture density network (TBMix).

Input 1: Invariants of Sij, Rij, gradp and gradk.
Input 2: Tensor basis of Sij and Rij.
Sij = Mean strain rate tensor, Rij = Mean rotation rate tensor (both Sij and Rij are
non-dimensionalised by k/eps). gradp = pressure gradient, gradk = TKE gradient.
Sij, Rij, gradp and gradk are from RANS simulations.

Output: Reynolds stress anisotropy tensor bij.
True bij can be from highly-resolved LES or DNS or experimental results. The TBMix aims to
produce this true output.

Reference for data driven turbulence modelling implementation:
Ling, J., Kurzawski, A. and Templeton, J., 2016. Reynolds averaged turbulence modelling
using deep neural networks with embedded invariance. Journal of Fluid Mechanics, 807,
pp.155-166.

Reference for mixture density networks:
Bishop, C.M., 1994. Mixture Density Networks. Technical report NCRG/94/004,
Aston University, Birmingham, UK.

Checked 24/05/2024 [✓]
Debugged 13/06/2024 [✓]
"""

import numpy as np
import timeit
import sys

sys.path.insert(1, "../TBNN")
from TBNN import case_dicts, results_writer
from TBNN.pred_iterator import preprocessing
from tbmix_pred_iterator import trial_iter


def tbmix_main(database, case_dict, incl_zonal_markers=False, num_zonal_markers=0):
    # Define parameters, d = default
    num_kernels = 3  # Number of kernels
    num_hid_layers = 5  # Number of hidden layers, d = 3
    num_hid_nodes = [5, 10, 15, 20, 7]  # [10] * num_hid_layers  # Number of nodes in
    # each hidden layer given as a vector, d = [5, 5, 5]

    num_tensor_basis = 3  # Number of tensor bases for bij pred; for 2D = 3, for 3D = 10
    max_epochs = 20  # Max number of training epochs, d = 2000
    min_epochs = 15  # Min number of training epochs, d = 1000
    interval = 1  # Frequency of epochs at which the model is validated, d = 100
    avg_interval = 2  # Number of intervals averaged over for early stop criteria, d = 4
    enforce_realiz = True  # Enforce realizability on Reynolds stresses, d = True
    num_realiz_its = 5  # Number of iterations for enforcing realizability, d = 5

    # Define advanced parameters, d = default
    af = ["ReLU"] * num_hid_layers  # Activation functions, d = ["ELU"]*num_hid_layers
    af_params = None  # Parameters of the activation functions,
    # d = ["alpha=1.0, inplace=False"]*num_hid_layers

    weight_init = "kaiming_normal_"  # Weight initialiser, d = "kaiming_normal_"
    weight_init_params = "nonlinearity=leaky_relu"  # Weight initialiser arguments,
    # d = "nonlinearity=leaky_relu"

    init_lr = 1e-4  # Initial learning rate, d = 0.01
    lr_scheduler = "ExponentialLR"  # Learn rate scheduler, d = ExponentialLR
    lr_scheduler_params = "gamma=1"  # Learn rate scheduler parameters, d = "gamma=0.9"
    loss = "nllLoss"  # Loss function, d = "nllLoss"
    optimizer = "Adam"  # Optimizer, d = "Adam"
    batch_size = 32  # Training batch size, d = 16

    # Define TBMix inputs
    two_invars = True  # Only include the first two invariants tr(S²) and tr(R²)
    incl_p_invars = False  # Include pressure invariants in inputs
    incl_tke_invars = False  # Include tke invariants in inputs
    incl_input_markers = False  # Include scalar markers in inputs
    num_input_markers = None  # Number of scalar markers in inputs
    rho = 1.514  # Density of air at -40C with kinematic viscosity, nu = 1e-5 m²/s

    # Define splitting of training, validation and testing datasets
    train_test_rand_split = False  # Randomly split database for train and test, d = False
    train_test_split_frac = None  # Fraction of database used for training and
    # validation if train_test_rand_split = True, d = 0
    train_valid_rand_split = False  # Randomly split training database for training and
    # validation, d = False
    train_valid_split_frac = None  # Fraction of training database used for training while
    # the other fraction is used for validation if train_valid_rand_split = True, d = 0

    # Set case lists if train_test_rand_split and train_valid_rand_split = False
    train_list = [0.5, 1.5]
    valid_list = [1]
    test_list = [1.2]

    # Other
    num_dims = 2  # Number of dimensions in dataset, d = 2
    num_seeds = 1  # Number of reproducible TBMix predictions to save
    print_freq = 1  # Console print frequency

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    folder_path = results_writer.create_parent_folders()  # ✓
    start = timeit.default_timer()

    coords, x, tb, y, num_inputs = \
        preprocessing(database, num_dims, num_input_markers, num_zonal_markers,
                      two_invars, incl_p_invars, incl_tke_invars, incl_input_markers,
                      incl_zonal_markers, rho, num_tensor_basis, enforce_realiz,
                      num_realiz_its)  # ✓
    user_vars = locals()
    current_folder = trial_iter(num_seeds, coords, x, tb, y, train_list, valid_list,
                                test_list, train_valid_rand_split, train_valid_split_frac,
                                train_test_rand_split, train_test_split_frac,
                                num_tensor_basis, num_hid_layers, num_hid_nodes, af,
                                af_params, init_lr, lr_scheduler, lr_scheduler_params,
                                weight_init, weight_init_params, max_epochs,
                                min_epochs, interval, avg_interval, loss, optimizer,
                                batch_size, enforce_realiz, num_realiz_its, folder_path,
                                user_vars, print_freq, case_dict, num_inputs,
                                num_kernels)  # ✓

    stop = timeit.default_timer()
    results_writer.write_time(start, stop, folder_path, current_folder)  # ✓
    print("Finished running TBMix training, validation, and testing")


if __name__ == "__main__":
    # Load database and associated dictionary ✓
    database_name = "PHLL4_dataset.txt"  # Data source
    db2case = case_dicts.case_dict_names()
    case_dict, _, num_skip_rows = db2case[database_name]  # ✓
    database = np.loadtxt(database_name, skiprows=num_skip_rows)

    # Run TBMix
    tbmix_main(database, case_dict)  #
