"""
This program performs all the steps for training and testing a FF-FC-NN for TKE prediction
"""

import numpy as np
import timeit
import sys

from tke_core import tkenn_ops
from tke_preprocessor import calc_output, split_database
from tke_results_writer import write_param_txt, write_param_csv, write_test_truth_logk, write_k_results
from TBNN import calculator, case_dicts, pred_iterator, preprocessor, results_writer


def tkenn_main(database, case_dict, incl_zonal_markers=False, num_zonal_markers=0):
    # Define parameters
    num_hid_layers = 3  # Number of hidden layers in the TBNN, default = 2
    num_hid_nodes = [7, 8, 2]  # Number of nodes in the hidden layers given as a vector, default = [5, 5, 5, 5, 5]
    max_epochs = 50  # Max number of training epochs, default = 2000
    min_epochs = 10  # Min number of training epochs, default = 1000
    interval = 2  # Frequency at which convergence is checked, default = 100
    avg_interval = 3  # Number of intervals averaged over for early stopping criteria, default = 4

    # Define advanced parameters
    af = ["ELU"]*3
    af_params = ["alpha=0.8"]*3
    weight_init = "xavier_uniform_"  # Weight initialiser algorithm, default = "xavier_uniform"
    weight_init_params = "gain=1"  # Arguments of the weight initialiser algorithm, default = "gain=1.0"
    init_lr = 0.001  # Initial learning rate, default = 0.01
    lr_scheduler = "ExponentialLR"  # Learning rate scheduler, default = ExponentialLR
    lr_scheduler_params = "gamma=0.1"  # Parameters of learning rate scheduler, default = "gamma=0.1"
    loss = "MSELoss"  # Loss function, default = "MSELoss"
    optimizer = "Adam"  # Optimizer algorithm, default = "Adam"
    batch_size = 5  # Training batch size, default = 1

    # Define database and data splits for training, validation and testing
    incl_p_invars = False
    incl_tke_invars = False
    incl_input_markers = False  # Include scalar markers in inputs
    num_input_markers = None  # Number of scalar markers in inputs
    rho = 1.514  # Density of air at -40C with nu = 1e-5 m^2/s

    train_test_rand_split = False  # Randomly split entire database for training and testing, default = False
    train_test_split_frac = 0  # Fraction of entire database used for training and validation if random split = True,
    # default = 0
    train_valid_rand_split = False  # Randomly split training database for training and validation, default = False
    train_valid_split_frac = 0  # Fraction of training database used for actual training, the other fraction is used
    # for validation, default = 0. Train-test split must be run before this.

    # Set case lists if train_test_rand_split and train_valid_rand_split = False
    train_list = [180, 290, 490, 760, 945]
    valid_list = [395]
    test_list = [590]

    # Other
    n_seeds = 5  # Number of reproducible TBNN predictions to save
    print_freq = 2  # Console print frequency

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    sys.path.insert(1, "../TBNN")

    # Load in data
    folder_path = results_writer.create_parent_folders()
    Cx, Cy, k, eps, grad_u, grad_p, u, grad_k, input_markers, tauij = \
        pred_iterator.load_data(database, num_input_markers, num_zonal_markers=num_zonal_markers,
                                pressure_tf=incl_p_invars, tke_tf=incl_tke_invars, input_markers_tf=incl_input_markers,
                                zonal_markers_tf=incl_zonal_markers)
    print("Data loading complete")

    # Calculate inputs and outputs
    data_processor = calculator.PopeDataProcessor()
    _, _, x = preprocessor.scalar_basis_manager(data_processor, k, eps, grad_u, rho, u, grad_p, grad_k,
                                                incl_p_invars=incl_p_invars, incl_tke_invars=incl_tke_invars)
    x = np.concatenate((x, input_markers), axis=1)
    num_inputs = x.shape[1]
    y = calc_output(tauij)  # y = log(k)
    print("x and y calculations complete")

    # Loop the following for each instance (seed) of TKENN:
    user_vars = locals()
    for seed in range(1, n_seeds + 1):
        # Initialise logs and results lists, write parameters and set random seed
        current_folder, log = results_writer.init_log(folder_path, seed)
        if seed == 1:
            final_train_rmse_list, final_valid_rmse_list, test_rmse_list = results_writer.errors_list_init()
            write_param_txt(current_folder, folder_path, user_vars)
            write_param_csv(current_folder, folder_path, user_vars)
        pred_iterator.set_seed(seed)

        # Split database into training, validation and testing datasets
        Cx_train, Cy_train, x_train, y_train, Cx_test, Cy_test, x_test, y_test, Cx_valid, Cy_valid, x_valid, y_valid = \
            split_database(Cx, Cy, x, y, train_list, valid_list, test_list, seed, train_valid_rand_split,
                           train_test_rand_split, train_valid_split_frac, train_test_split_frac, case_dict)
        if seed == 1:
            write_test_truth_logk(Cx_test, Cy_test, folder_path, y_test)

        # Perform TKENN training, validation and testing
        epoch_count, final_train_rmse, final_valid_rmse, y_pred, test_rmse = \
            tkenn_ops(x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size, num_hid_layers, num_hid_nodes, af,
                      af_params, seed, weight_init, weight_init_params, loss, optimizer, init_lr, lr_scheduler,
                      lr_scheduler_params, min_epochs, max_epochs, interval, avg_interval, print_freq, log, num_inputs)

        # Write prediction and append RMSE results
        write_k_results(Cx_test, Cy_test, folder_path, seed, y_pred, current_folder)
        final_train_rmse_list.append(final_train_rmse)
        final_valid_rmse_list.append(final_valid_rmse)
        test_rmse_list.append(test_rmse)

    # After all seeds:
    results_writer.write_trial_rmse_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list, folder_path,
                                        current_folder)
    results_writer.write_error_means_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list, folder_path,
                                         current_folder)

    return folder_path, current_folder


if __name__ == "__main__":
    start = timeit.default_timer()

    # Load database and associated dictionary
    database_name = "full_set_no_walls.txt"  # Data source
    db2case = case_dicts.case_dict_names()
    case_dict, _, num_skip_rows = db2case[database_name]
    database = np.loadtxt(database_name, skiprows=num_skip_rows)

    # Run TKENN
    folder_path, current_folder = tkenn_main(database, case_dict)
    stop = timeit.default_timer()
    results_writer.write_time(start, stop, folder_path, current_folder)
    print("TKENN finished")
