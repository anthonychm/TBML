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

import TBNN as tbnn
import tbmix_pred_iterator as tbmix_piter
import numpy as np
import timeit
import sys
sys.path.append('../TBNN')


def tbmix_main(database, case_dict, incl_zonal_markers=False, num_zonal_markers=0):
    # Define parameters (d = default)
    num_kernels = 3  # Num. of kernels
    num_hid_layers = 5  # Num. of hidden layers, d = 3
    num_hid_nodes = [5, 10, 15, 20, 7]  # Num. of hidden nodes, d = [5, 5, 5]
    max_epochs = 10000  # Max num. of epochs, d = 2000
    min_epochs = 50  # Min num. of epochs, d = 1000
    interval = 5  # Model undertakes validation after every interval of epochs, d = 100
    avg_interval = 3  # Num. of intervals averaged over for early stopping, d = 4
    enforce_realiz = True  # Enforce realizability on Reynolds stresses, d = True
    num_realiz_its = 5  # Num. of iterations for realizability enforcing, d = 5

    # Define advanced parameters
    af = ["ReLU"] * num_hid_layers  # Activation functions, d = ["ELU"]*num_hid_layers
    af_params = None  # Activation function parameters, d = None
    weight_init = "kaiming_normal_"  # Weight initialiser, d = "kaiming_normal_"
    weight_init_params = "nonlinearity=leaky_relu"  # Weight initialiser params, d = "nonlinearity=leaky_relu"
    init_lr = 1e-4  # Initial learning rate, d = 0.01
    lr_scheduler = "ExponentialLR"  # Learning rate scheduler, d = "ExponentialLR"
    lr_scheduler_params = "gamma=1"  # Learning rate scheduler params, d = "gamma=0.9"
    loss = "nllLoss"  # Loss function, d = "nllLoss"
    optimizer = "Adam"  # Optimizer, d = "Adam"
    batch_size = 32  # Batch size, d = 16

    # Define TBMix inputs
    two_invars = True  # Only include the first two invariants tr(S²) and tr(R²)
    incl_p_invars = False  # Include pressure invariants in inputs
    incl_tke_invars = False  # Include tke invariants in inputs
    incl_input_markers = False  # Include scalar markers in inputs
    num_input_markers = None  # Number of scalar markers in inputs
    rho = 1.514  # Density of air at -40C with kinematic viscosity, nu = 1e-5 m²/s
    num_tensor_basis = 3  # Num. of tensor bases; for 2D flow = 3, for 3D flow = 10

    # Define splitting of training, validation and testing datasets
    train_test_rand_split = False  # Randomly split database for train and test, d = False
    train_test_split_frac = None  # Fraction of database used for training and
    # validation if train_test_rand_split = True, d = 0
    train_valid_rand_split = False  # Randomly split training database for training and
    # validation, d = False
    train_valid_split_frac = None  # Fraction of training database used for training while
    # the other fraction is used for validation if train_valid_rand_split = True, d = 0

    # Set case lists if train_test_rand_split and train_valid_rand_split = False
    train_list = [180, 490, 590, 945]  # [0.5, 1.5]
    valid_list = [290, 760]  # [1]
    test_list = [395]  # [1.2]

    # Other
    num_dims = 1  # Num. of dimensions in dataset, d = 2
    num_seeds = 1  # Num. of reproducible TBMix predictions to save
    print_freq = 1  # Print frequency to console

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    folder_path = tbnn.write.create_parent_folders()  # ✓
    start = timeit.default_timer()
    assert two_invars is True  # Code only currently works for 2D tensor basis
    assert num_tensor_basis == 3  # Code only currently works for 2D tensor basis

    coords, x, tb, y, num_inputs = \
        tbnn.piter.preprocessing(database, num_dims, num_input_markers,
                                 num_zonal_markers, two_invars, incl_p_invars,
                                 incl_tke_invars, incl_input_markers,
                                 incl_zonal_markers, rho, num_tensor_basis,
                                 enforce_realiz, num_realiz_its)  # ✓
    user_vars = locals()
    current_folder = \
        tbmix_piter.trial_iter(num_seeds, coords, x, tb, y, train_list, valid_list,
                               test_list, train_valid_rand_split,
                               train_valid_split_frac, train_test_rand_split,
                               train_test_split_frac, num_tensor_basis, num_hid_layers,
                               num_hid_nodes, af, af_params, init_lr, lr_scheduler,
                               lr_scheduler_params, weight_init, weight_init_params,
                               max_epochs, min_epochs, interval, avg_interval, loss,
                               optimizer, batch_size, enforce_realiz, num_realiz_its,
                               folder_path, user_vars, print_freq, case_dict,
                               num_inputs, num_kernels)  # ✓

    stop = timeit.default_timer()
    tbnn.write.write_time(start, stop, folder_path, current_folder)  # ✓
    print("Finished running TBMix training, validation, and testing")


if __name__ == "__main__":
    # Load database and associated dictionary ✓
    database_name = "CHAN7_database.txt"  # Data source
    db2case = tbnn.case_dicts.case_dict_names()
    case_dict, _, num_skip_rows = db2case[database_name]  # ✓
    database = np.loadtxt(database_name, skiprows=num_skip_rows)

    # Run TBMix
    tbmix_main(database, case_dict)  #
