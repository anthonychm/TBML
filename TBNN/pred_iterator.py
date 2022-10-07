from preprocessor import load_data
from results_writer import write_bij_results, init_log, errors_list_init, write_error_means_csv, write_trial_rmse_csv, \
    write_test_truth_bij, write_param_txt, write_param_csv
from calculator import PopeDataProcessor
import case_dicts
import torch
import numpy as np
import random
from core import NetworkStructure, Tbnn, DataLoader, TbnnTVT


def preprocessing(enforce_realiz, num_realiz_its, database_name, n_skiprows, num_tensor_basis):
    """
    Preprocess the CFD data, which consists of the following steps:
    1) Load in data using load_channel_data function
    2) Calculate Sij and Rij for the RANS data using calc_Sij_Rij function
    3) Calculate invariants for the RANS data using calc_scalar_basis function
    4) Calculate tensor basis for the RANS data using calc_tensor_basis function
    5) Calculate Reynolds stress anisotropy for the LES/DNS/experimental data using calc_output function
    6) If enforce_realizability is True, enforce realizability in y using make_realizable function
    :param enforce_realiz: Boolean for enforcing realizability in y
    :param num_realiz_its: Number of iterations for enforcing realizability in y
    :param database_name: Name of the data file containing the CFD data
    :param n_skiprows: Number of rows at top of data file which this code should skip for reading
    :return: invariants x, tensor basis tb and Reynolds stress anisotropy y, for all of the CFD data
    """

    # Load in data
    k, eps, grad_u, stresses = load_data(database_name=database_name, n_skiprows=n_skiprows)
    print("Data loading complete")

    # Calculate inputs and outputs
    data_processor = PopeDataProcessor()
    Sij, Rij = data_processor.calc_Sij_Rij(grad_u, k, eps)  # Mean strain rate tensor k/eps*Sij,
                                                            # mean rotation rate tensor k/eps*Rij
    x = data_processor.calc_scalar_basis(Sij, Rij, is_train=True)  # Scalar basis # to do: Check this
    tb = data_processor.calc_tensor_basis(Sij, Rij, num_tensor_basis)  # Tensor basis
    y = data_processor.calc_output(stresses)  # Anisotropy tensor
    print("x, tb and y calculations complete")

    # Enforce realizability
    if enforce_realiz is True:
        for i in range(num_realiz_its):
            y = PopeDataProcessor.make_realizable(y)
        print("Realizability enforced")

    return x, tb, y


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return


def split_database(x, tb, y, train_list, valid_list, test_list, seed, train_valid_rand_split, train_test_rand_split,
                   train_valid_split_frac, train_test_split_frac, num_tensor_basis, database_name):
    """
    Split the data using your chosen data-splitting method
    :param x: scalar invariants
    :param tb: tensor basis
    :param y: Reynolds stress anisotropy
    :param seed: Current random seed for reproducible database splitting
    :param train_list: list of cases for training (use if random_split = False)
    :param test_list: list of cases for testing (use if random_split = False)
    :param train_test_rand_split: Boolean for enforcing random train-test splitting of data (True) or specifying certain cases
    for training and testing (False)
    :param train_test_split_frac: fraction of all CFD data to use for training data (use if random_split = True)
    :return: inputs, tb and outputs for the training and test sets
    """

    # Split database randomly if random split = True
    if train_test_rand_split is True:
        x_train, tb_train, y_train, x_test, tb_test, y_test = \
            PopeDataProcessor.random_split(x, tb, y, fraction=train_test_split_frac, seed=seed)
        print("Train-test random data split complete")
        if train_valid_rand_split is True:
            x_train, tb_train, y_train, x_valid, tb_valid, y_valid = \
                PopeDataProcessor.random_split(x_train, tb_train, y_train, fraction=train_valid_split_frac, seed=seed)
            print("Train-valid random data split complete")
        else:
            x_valid, tb_valid, y_valid = [], [], []
            print("No data allocated for validation")

    # Else split database according to specified cases if random split = False
    else:
        db2case = case_dicts.case_dict_names()
        case_dict, n_case_points = db2case[database_name]
        x_train, tb_train, y_train = PopeDataProcessor.specified_split(x, tb, y, train_list, n_case_points, case_dict, num_tensor_basis, seed)
        print("Specified data split for training complete")
        x_test, tb_test, y_test = PopeDataProcessor.specified_split(x, tb, y, test_list, n_case_points, case_dict, num_tensor_basis, seed)
        print("Specified data split for testing complete")
        if train_valid_rand_split is False:
            x_valid, tb_valid, y_valid = PopeDataProcessor.specified_split(x, tb, y, valid_list, n_case_points, case_dict, num_tensor_basis, seed)
            print("Specified data split for validation complete")
        else:
            x_valid, tb_valid, y_valid = [], [], []
            print("No data allocated for validation")

    return x_train, tb_train, y_train, x_test, tb_test, y_test, x_valid, tb_valid, y_valid


def tbnn_ops(x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test, tb_test, y_test, batch_size, num_hid_layers,
             num_hid_nodes, af, af_params, seed, weight_init, weight_init_params, loss, optimizer, init_lr, lr_scheduler, lr_scheduler_params, min_epochs,
              max_epochs, interval, avg_interval, print_freq, log, enforce_realiz, num_realiz_its, num_tensor_basis):

    # Use GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare batch data in dataloaders and check dataloaders
    dataloader = DataLoader(x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test, tb_test, y_test, batch_size)
    DataLoader.check_data_loaders(dataloader.train_loader, x_train, tb_train, y_train, num_inputs=5,
                                  num_tensor_basis=num_tensor_basis)
    DataLoader.check_data_loaders(dataloader.valid_loader, x_valid, tb_valid, y_valid, num_inputs=5,
                                  num_tensor_basis=num_tensor_basis)
    DataLoader.check_data_loaders(dataloader.test_loader, x_test, tb_test, y_test, num_inputs=5,
                                  num_tensor_basis=num_tensor_basis)

    # Prepare TBNN architecture structure and check structure
    structure = NetworkStructure(num_hid_layers, num_hid_nodes, af, af_params, num_inputs=5,
                                 num_tensor_basis=num_tensor_basis)
    structure.check_structure()
    print("TBNN architecture structure and data loaders checked")

    # Construct TBNN and perform training, validation and testing
    tbnn = Tbnn(device, seed, structure, weight_init=weight_init, weight_init_params=weight_init_params).double()
    tbnn_tvt = TbnnTVT(loss, optimizer, init_lr, lr_scheduler, lr_scheduler_params, min_epochs, max_epochs, interval,
                       avg_interval, print_freq, log, model=tbnn)
    epoch_count, final_train_rmse, final_valid_rmse = tbnn_tvt.fit(device, train_loader=dataloader.train_loader,
                                                                   valid_loader=dataloader.valid_loader, model=tbnn)
    y_pred, test_rmse = tbnn_tvt.perform_test(device, enforce_realiz, num_realiz_its, log,
                                              test_loader=dataloader.test_loader, model=tbnn)

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


def trial_iter(n_seeds, x, tb, y, train_list, valid_list, test_list, train_valid_rand_split, train_valid_split_frac,
               train_test_rand_split, train_test_split_frac, num_tensor_basis, num_hid_layers, num_hid_nodes, af,
               af_params, init_lr, lr_scheduler, lr_scheduler_params, weight_init, weight_init_params, max_epochs,
               min_epochs, interval, avg_interval, loss, optimizer, batch_size, enforce_realiz, num_realiz_its,
               folder_path, user_vars, print_freq, database_name):

    for seed in range(1, n_seeds+1):
        current_folder, log = init_log(folder_path, seed)
        if seed == 1:
            final_train_rmse_list, final_valid_rmse_list, test_rmse_list = errors_list_init()
            write_param_txt(current_folder, folder_path, user_vars)
            write_param_csv(current_folder, folder_path, user_vars)
        set_seed(seed)
        x_train, tb_train, y_train, x_test, tb_test, y_test, x_valid, tb_valid, y_valid = \
            split_database(x=x, tb=tb, y=y, train_list=train_list, valid_list=valid_list, test_list=test_list,
                           seed=seed, train_valid_rand_split=train_valid_rand_split,
                           train_test_rand_split=train_test_rand_split, train_valid_split_frac=train_valid_split_frac,
                           train_test_split_frac=train_test_split_frac, num_tensor_basis=num_tensor_basis,
                           database_name=database_name)
        write_test_truth_bij(folder_path, seed, true_bij=y_test)
        epoch_count, final_train_rmse, final_valid_rmse, y_pred, test_rmse, = \
            tbnn_ops(x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test, tb_test, y_test, batch_size,
                      num_hid_layers, num_hid_nodes, af, af_params, seed, weight_init, weight_init_params, loss,
                      optimizer, init_lr, lr_scheduler, lr_scheduler_params, min_epochs, max_epochs, interval, avg_interval, print_freq, log,
                      enforce_realiz, num_realiz_its, num_tensor_basis)
        write_bij_results(folder_path, seed, y_pred, current_folder)
        # del x_train, tb_train, y_train, x_test, tb_test, y_test, x_valid, tb_valid, y_valid, y_pred
        final_train_rmse_list.append(final_train_rmse)
        final_valid_rmse_list.append(final_valid_rmse)
        test_rmse_list.append(test_rmse)

    # After all seeds:
    write_trial_rmse_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list, folder_path, current_folder)
    write_error_means_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list, folder_path, current_folder)

    return current_folder
