"""
===============================================
===== This code was written by Anthony Man ====
====== The University of Manchester, 2022 =====
===============================================

This program performs all the steps for training and testing a FF-FC-NN for turbulent
kinetic energy prediction.

Inputs: Invariants of Sij, Rij, gradp and gradk.
Sij = Mean strain rate tensor, Rij = Mean rotation rate tensor (both Sij and Rij are
non-dimensionalised by k/eps). gradp = pressure gradient, gradk = TKE gradient.
Sij, Rij, gradp and gradk are from RANS simulations.
Output: Turbulent kinetic energy (TKE).

Syntax and run checked by AM, 21/12/2022
PoF Oct 2022 models run using:
Local: Python 3.10.9, numpy 1.23.5, torch 1.13.0 and pandas 1.5.2
CSF4: Python 3.10.4, numpy 1.22.3, torch 1.13.1 and pandas 1.4.2
"""

import numpy as np
import timeit
import sys

from tke_core import tkenn_ops
from tke_preprocessor import calc_output, split_database
from tke_results_writer import write_param_txt, write_param_csv, write_test_truth_logk, \
    write_k_results

sys.path.insert(1, "../TBNN")
from TBNN import calculator, case_dicts, pred_iterator, preprocessor, results_writer


def tkenn_main(database, case_dict, incl_zonal_markers=False, num_zonal_markers=0):
    # Define parameters
    num_hid_layers = 5  # Number of hidden layers in the TKENN, default = 2
    num_hid_nodes = [10] * num_hid_layers  # Number of nodes in the hidden layers given
    # as a vector, default = [5, 5, 5]
    max_epochs = 100000  # Max number of training epochs, default = 2000
    min_epochs = 25  # Min number of training epochs, default = 1000
    interval = 4  # Frequency of epochs at which the model is evaluated on validation
    # dataset, default = 100
    avg_interval = 3  # Number of intervals averaged over for early stopping criteria,
    # default = 4

    # Define advanced parameters
    af = ["ELU"] * num_hid_layers  # Nonlinear activation function, default =
    # ["ELU"] * num_hid_layers
    af_params = ["inplace=False"] * num_hid_layers  # Parameters of the nonlinear
    # activation function, default = ["inplace=False"] * num_hid_layers
    weight_init = "kaiming_normal_"  # Weight initialiser algorithm,
    # default = "kaiming_normal_"
    weight_init_params = "nonlinearity=leaky_relu"  # Arguments of the weight
    # initialiser algorithm, default = "nonlinearity=leaky_relu"
    init_lr = 0.01  # Initial learning rate, default = 0.01
    lr_scheduler = "ExponentialLR"  # Learning rate scheduler, default = "ExponentialLR"
    lr_scheduler_params = "gamma=0.98"  # Parameters of learning rate scheduler,
    # default = "gamma=0.9"
    loss = "MSELoss"  # Loss function, default = "MSELoss"
    optimizer = "Adam"  # Optimizer algorithm, default = "Adam"
    batch_size = [256]  # [16, 32, 64, 128, 256]  # Training batch size, default = 16

    # Define TKENN inputs
    incl_RANS_k = True  # Include RANS k in inputs
    incl_p_invars = False  # Include pressure invariants in inputs
    incl_tke_invars = False  # Include tke invariants in inputs
    incl_input_markers = False  # Include scalar markers in inputs
    num_input_markers = None  # Number of scalar markers in inputs
    rho = 1.514  # Density of air at -40C with nu = 1e-5 m^2/s

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
    folder_path = results_writer.create_parent_folders()

    for b in batch_size:
        start = timeit.default_timer()

        # Load in data ✓
        coords, k, eps, grad_u, grad_p, u, grad_k, input_markers, tauij = \
            preprocessor.load_data(database, num_dims, num_input_markers,
                                   num_zonal_markers, incl_p_invars, incl_tke_invars,
                                   incl_input_markers, incl_zonal_markers)  # ✓
        print("Data loading complete")

        # Calculate inputs and outputs ✓
        data_processor = calculator.PopeDataProcessor()  # ✓
        _, _, x = preprocessor.scalar_basis_manager(data_processor, k, eps, grad_u, rho,
                                                    u, grad_p, grad_k, incl_p_invars,
                                                    incl_tke_invars)  # ✓
        if incl_input_markers is True:
            x = np.concatenate((x, input_markers), axis=1)  # ✓
        if incl_RANS_k is True:
            # Log-normalize k
            x = np.concatenate((x, np.log10(np.expand_dims(k, axis=1))), axis=1)
        num_inputs = x.shape[1]
        y = calc_output(tauij)  # y = log(k) ✓
        print("x and y calculations complete")
        user_vars = locals()

        # Loop the following for each instance (seed) of TKENN:
        for seed in range(1, num_seeds+1):
            # Set up results logs, lists and files ✓
            current_folder, log = results_writer.init_log(folder_path, seed)  # ✓
            if seed == 1:
                final_train_rmse_list, final_valid_rmse_list, test_rmse_list = \
                    results_writer.errors_list_init()  # ✓
                write_param_txt(current_folder, folder_path, user_vars)  # ✓
                write_param_csv(current_folder, folder_path, user_vars)  # ✓

            # Prepare TVT datasets ✓
            pred_iterator.set_seed(seed)  # ✓
            coords_train, x_train, y_train, coords_test, x_test, y_test, coords_valid, \
                x_valid, y_valid = \
                split_database(coords, x, y, train_list, valid_list, test_list, seed,
                               train_valid_rand_split, train_test_rand_split,
                               train_valid_split_frac, train_test_split_frac,
                               case_dict)  # ✓
            if seed == 1:
                write_test_truth_logk(coords_test, folder_path, y_test)  # ✓

            # Run TKENN operations ✓
            epoch_count, final_train_rmse, final_valid_rmse, y_pred, test_rmse = \
                tkenn_ops(x_train, y_train, x_valid, y_valid, x_test, y_test, b,
                          num_hid_layers, num_hid_nodes, af, af_params, seed,
                          weight_init, weight_init_params, loss, optimizer, init_lr,
                          lr_scheduler, lr_scheduler_params, min_epochs, max_epochs,
                          interval, avg_interval, print_freq, log, num_inputs)  # ✓

            # Write results for each seed ✓
            write_k_results(coords_test, folder_path, seed, y_pred, current_folder)  # ✓
            final_train_rmse_list.append(final_train_rmse)
            final_valid_rmse_list.append(final_valid_rmse)
            test_rmse_list.append(test_rmse)

        # Write results for each trial ✓
        results_writer.write_trial_rmse_csv(final_train_rmse_list, final_valid_rmse_list,
                                            test_rmse_list, folder_path,
                                            current_folder)  # ✓
        results_writer.write_error_means_csv(final_train_rmse_list, final_valid_rmse_list,
                                             test_rmse_list, folder_path,
                                             current_folder)  # ✓

        # Write running time ✓
        stop = timeit.default_timer()
        results_writer.write_time(start, stop, folder_path, current_folder)  # ✓
        print("TKENN finished")


if __name__ == "__main__":

    # Load database and associated dictionary ✓
    database_name = "FBFS5_full_set_no_walls.txt"  # Data source
    db2case = case_dicts.case_dict_names()
    case_dict, _, num_skip_rows = db2case[database_name]  # ✓
    database = np.loadtxt(database_name, skiprows=num_skip_rows)

    # Run TKENN ✓
    tkenn_main(database, case_dict)
