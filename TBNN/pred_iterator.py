from calculator import PopeDataProcessor
from core import NetworkStructure, Tbnn, DataLoader, TbnnTVT
from preprocessor import load_data, scalar_basis_manager, DataSplitter
from results_writer import write_bij_results, init_log, errors_list_init, write_error_means_csv, write_trial_rmse_csv, \
    write_test_truth_bij, write_param_txt, write_param_csv

import torch
import numpy as np
import random


def preprocessing(database, num_input_markers, num_zonal_markers, incl_p_invars, incl_tke_invars, incl_input_markers,
                  incl_zonal_markers, rho, num_tensor_basis, enforce_realiz, num_realiz_its):
    """
    Preprocess the CFD data:
    1) Load in data using load_data function
    2) Calculate Sij and Rij for the RANS data using calc_Sij_Rij function
    3) Calculate Sij and Rij invariants for the RANS data using calc_scalar_basis function
    4) Calculate tensor basis for the RANS data using calc_tensor_basis function
    5) Calculate Reynolds stress anisotropy bij for the LES/DNS/experimental data using calc_output function
    6) If enforce realizability is True, enforce realizability in bij using make_realizable function

    :param enforce_realiz: boolean for enforcing realizability in y
    :param num_realiz_its: number of iterations for enforcing realizability in y
    :param num_tensor_basis: number of tensor basis for bij prediction

    :return: invariants x, tensor basis tb and Reynolds stress anisotropy y, for all of the CFD data
    """

    # Load in data
    Cx, Cy, k, eps, grad_u, grad_p, u, grad_k, input_markers, tauij = \
        load_data(database, num_input_markers, num_zonal_markers=num_zonal_markers, pressure_tf=incl_p_invars,
                  tke_tf=incl_tke_invars, input_markers_tf=incl_input_markers, zonal_markers_tf=incl_zonal_markers)
    print("Data loading complete")

    # Calculate inputs and outputs
    data_processor = PopeDataProcessor()
    Sij, Rij, x = scalar_basis_manager(data_processor, k, eps, grad_u, rho, u, grad_p, grad_k,
                                       incl_p_invars=incl_p_invars, incl_tke_invars=incl_tke_invars)
    num_inputs = x.shape[1]
    tb = data_processor.calc_tensor_basis(Sij, Rij, num_tensor_basis)  # Tensor basis
    y = data_processor.calc_output(tauij)  # Anisotropy tensor bij
    print("x, tb and y calculations complete")

    # Enforce realizability
    if enforce_realiz is True:
        for i in range(num_realiz_its):
            y = data_processor.make_realizable(y)
        print("Realizability enforced")

    return Cx, Cy, x, tb, y, num_inputs


def set_seed(seed):
    """
    Set random seeds for all operations with seed parameter. This ensures the results are reproducible.
    :param seed: integer between 1 and the value of maximum seed runs
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def split_database(Cx, Cy, x, tb, y, train_list, valid_list, test_list, seed, train_valid_rand_split,
                   train_test_rand_split, train_valid_split_frac, train_test_split_frac, num_tensor_basis, case_dict):
    """
    Split data using the random or specified data-splitting method. The random method randomly allocates (x, tb, y)
    data for TVT. The specified method allocates data from specific cases for TVT according to the TVT lists.
    TVT = training, validation and testing.

    :param x: scalar invariants of Sij and Rij
    :param tb: tensor basis
    :param y: Reynolds stress anisotropy
    :param train_list: list of cases for training (use if train_test_rand_split = False)
    :param valid_list: list of cases for validation (use if train_valid_rand_split = False)
    :param test_list: list of cases for testing (use if train_test_rand_split = False)
    :param seed: current random seed for reproducible database splitting
    :param train_valid_rand_split: boolean for enforcing random train-valid splitting of training data
    :param train_test_rand_split: boolean for enforcing random train-test splitting of all data
    :param train_valid_split_frac: fraction of training data to use for actual training,
    the rest is used for validation (use if train_valid_rand_split = True)
    :param train_test_split_frac: fraction of all data to use for training data (use if train_test_rand_split = True)
    :param num_tensor_basis: number of tensor basis for bij prediction

    :return: inputs, tensor basis and outputs for the TVT data sets
    """

    # Split database randomly if random split = True
    if train_test_rand_split is True:
        Cx_train, Cy_train, x_train, tb_train, y_train, Cx_test, Cy_test, x_test, tb_test, y_test = \
            DataSplitter.random_split(Cx, Cy, x, tb, y, train_test_split_frac, seed)
        print("Train-test random data split complete")
        if train_valid_rand_split is True:
            Cx_train, Cy_train, x_train, tb_train, y_train, Cx_valid, Cy_valid, x_valid, tb_valid, y_valid = \
                DataSplitter.random_split(Cx, Cy, x_train, tb_train, y_train, train_valid_split_frac, seed)
            print("Train-valid random data split complete")
        else:
            Cx_valid, Cy_valid, x_valid, tb_valid, y_valid = [], [], [], [], []
            print("No data allocated for validation")

    # Else split database according to specified cases if random split = False
    else:
        Cx_train, Cy_train, x_train, tb_train, y_train = \
            DataSplitter.specified_split(Cx, Cy, x, tb, y, train_list, case_dict, num_tensor_basis, seed)
        print("Specified data split for training complete")
        Cx_test, Cy_test, x_test, tb_test, y_test = \
            DataSplitter.specified_split(Cx, Cy, x, tb, y, test_list, case_dict, num_tensor_basis, seed, shuffle=False)
        print("Specified data split for testing complete")
        if train_valid_rand_split is False:
            Cx_valid, Cy_valid, x_valid, tb_valid, y_valid = \
                DataSplitter.specified_split(Cx, Cy, x, tb, y, valid_list, case_dict, num_tensor_basis, seed,
                                             shuffle=False)
            print("Specified data split for validation complete")
        else:
            Cx_valid, Cy_valid, x_valid, tb_valid, y_valid = [], [], [], [], []
            print("No data allocated for validation")

    return Cx_train, Cy_train, x_train, tb_train, y_train, Cx_test, Cy_test, x_test, tb_test, y_test, \
        Cx_valid, Cy_valid, x_valid, tb_valid, y_valid


def tbnn_ops(x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test, tb_test, y_test, batch_size, num_hid_layers,
             num_hid_nodes, af, af_params, seed, weight_init, weight_init_params, loss, optimizer, init_lr, lr_scheduler, lr_scheduler_params, min_epochs,
              max_epochs, interval, avg_interval, print_freq, log, enforce_realiz, num_realiz_its, num_tensor_basis, num_inputs):

    # Use GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare batch data in dataloaders and check dataloaders
    dataloader = DataLoader(x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test, tb_test, y_test, batch_size)
    DataLoader.check_data_loaders(dataloader.train_loader, x_train, tb_train, y_train, num_inputs, num_tensor_basis)
    DataLoader.check_data_loaders(dataloader.valid_loader, x_valid, tb_valid, y_valid, num_inputs, num_tensor_basis)
    DataLoader.check_data_loaders(dataloader.test_loader, x_test, tb_test, y_test, num_inputs, num_tensor_basis)

    # Prepare TBNN architecture structure and check structure
    structure = NetworkStructure(num_hid_layers, num_hid_nodes, af, af_params, num_inputs, num_tensor_basis)
    structure.check_structure()
    print("TBNN architecture structure and data loaders checked")

    # Construct TBNN and perform training, validation and testing
    tbnn = Tbnn(device, seed, structure, weight_init=weight_init, weight_init_params=weight_init_params).double()
    tbnn_tvt = TbnnTVT(loss, optimizer, init_lr, lr_scheduler, lr_scheduler_params, min_epochs, max_epochs, interval,
                       avg_interval, print_freq, log, tbnn)
    epoch_count, final_train_rmse, final_valid_rmse = tbnn_tvt.fit(device, dataloader.train_loader,
                                                                   dataloader.valid_loader, tbnn)
    y_pred, test_rmse = tbnn_tvt.perform_test(device, enforce_realiz, num_realiz_its, log, dataloader.test_loader, tbnn)

    return epoch_count, final_train_rmse, final_valid_rmse, y_pred, test_rmse


"""
For a specified range of random seeds, the function below called tbnn_iterate executes the following:
1) If seed == 1, the RMSE lists are initialised as empty by running errors_list_init()
2) Sets the output log file 
3) Splits all the data into training and test sets using the database_splitting function (see above for this function)
4) Runs tbnn_operations using the tbnn_operations function (see above for this function)
5) Writes TBNN-predicted results to the folder of choice using the write_results function 
(see turbulence_example_driver.py for this function)
6) Deletes the training and test sets so that this function can be looped from the beginning
7) Appends the RMSEs to their respective lists
8) Calculates the average of these RMSEs and writes them to Trial_parameters_and_means.csv by running 
write_error_means()
9) Writes all the RMSEs for each trial to csv by running write_trial_rmse_csv()
"""


def trial_iter(n_seeds, Cx, Cy, x, tb, y, train_list, valid_list, test_list, train_valid_rand_split, train_valid_split_frac,
               train_test_rand_split, train_test_split_frac, num_tensor_basis, num_hid_layers, num_hid_nodes, af,
               af_params, init_lr, lr_scheduler, lr_scheduler_params, weight_init, weight_init_params, max_epochs,
               min_epochs, interval, avg_interval, loss, optimizer, batch_size, enforce_realiz, num_realiz_its,
               folder_path, user_vars, print_freq, case_dict, num_inputs):
    """
    After obtaining x, tb and y from preprocessing, this function is run for a user-specified number of random seeds
    to train n_seed TBNN instances and obtain n_seed predictions.
    1) Create new trial folder. A trial is a particular set-up of parameter values on the frontend.py file.
    2) If seed = 1, initialise TVT error lists and write the parameter values.
    3) Set the seed for all operations.
    4)
    """
    for seed in range(1, n_seeds+1):
        current_folder, log = init_log(folder_path, seed)
        if seed == 1:
            final_train_rmse_list, final_valid_rmse_list, test_rmse_list = errors_list_init()
            write_param_txt(current_folder, folder_path, user_vars)
            write_param_csv(current_folder, folder_path, user_vars)
        set_seed(seed)
        Cx_train, Cy_train, x_train, tb_train, y_train, Cx_test, Cy_test, x_test, tb_test, y_test, \
            Cx_valid, Cy_valid, x_valid, tb_valid, y_valid = \
            split_database(Cx, Cy, x, tb, y, train_list, valid_list, test_list, seed, train_valid_rand_split,
                           train_test_rand_split, train_valid_split_frac, train_test_split_frac, num_tensor_basis,
                           case_dict)
        if seed == 1:
            write_test_truth_bij(Cx_test, Cy_test, folder_path, y_test)
        epoch_count, final_train_rmse, final_valid_rmse, y_pred, test_rmse, = \
            tbnn_ops(x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test, tb_test, y_test, batch_size,
                      num_hid_layers, num_hid_nodes, af, af_params, seed, weight_init, weight_init_params, loss,
                      optimizer, init_lr, lr_scheduler, lr_scheduler_params, min_epochs, max_epochs, interval, avg_interval, print_freq, log,
                      enforce_realiz, num_realiz_its, num_tensor_basis, num_inputs)
        write_bij_results(Cx_test, Cy_test, folder_path, seed, y_pred, current_folder)
        final_train_rmse_list.append(final_train_rmse)
        final_valid_rmse_list.append(final_valid_rmse)
        test_rmse_list.append(test_rmse)

    # After all seeds:
    write_trial_rmse_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list, folder_path, current_folder)
    write_error_means_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list, folder_path, current_folder)

    return current_folder
