from calculator import PopeDataProcessor
from core import NetworkStructure, Tbnn, DataLoader, TbnnTVT
from preprocessor import load_data, scalar_basis_manager, DataSplitter
from results_writer import write_bij_results, init_log, errors_list_init, \
    write_error_means_csv, write_trial_rmse_csv, write_test_truth_bij, write_param_txt, \
    write_param_csv

import torch
import numpy as np
import random


def preprocessing(database, num_dims, num_input_markers, num_zonal_markers, two_invars,
                  incl_p_invars, incl_tke_invars, incl_input_markers, incl_zonal_markers,
                  rho, num_tensor_basis, enforce_realiz, num_realiz_its,
                  incl_nut_input=False):  # ✓
    """
    Preprocesses the CFD data:
    - Load and separate data using load_data
    - Calculate scalar invariants for the RANS data using scalar_basis_manager
    - Concatenate scalar invariants with input markers
    - Calculate tensor basis for the RANS data using calc_tensor_basis
    - Calculate anisotropy bij for the LES/DNS/experimental data using calc_output
    - If enforce_realiz is True, enforce realizability in bij using make_realizable

    :param database: Numpy array containing the full database
    :param num_dims: Number of coordinate dimensions in database
    :param num_input_markers: Number of input scalar markers
    :param num_zonal_markers: Number of zonal boundary markers
    :param incl_p_invars: Boolean for including invariants of pressure gradient
    :param incl_tke_invars: Boolean for including invariants of tke gradient
    :param incl_input_markers: Boolean for including input scalar markers
    :param incl_zonal_markers: Boolean for including zonal boundary markers
    :param rho: Fluid density for calculation of pressure gradient invariants
    :param num_tensor_basis: Number of tensor bases for bij prediction
    :param enforce_realiz: Boolean for enforcing realizability in bij truth data
    :param num_realiz_its: Number of iterations for enforcing realizability

    :return: Inputs x, tensor basis tb and Reynolds stress anisotropy y, for all the CFD
             data
    """

    # Load in data ✓
    coords, k, eps, grad_u, grad_p, u, grad_k, input_markers, tauij = \
        load_data(database, num_dims, num_input_markers, num_zonal_markers,
                  incl_p_invars, incl_tke_invars, incl_input_markers,
                  incl_zonal_markers)  # ✓
    print("Data loading complete")

    # Calculate inputs and outputs ✓
    data_processor = PopeDataProcessor()
    Sij, Rij, x = scalar_basis_manager(data_processor, k, eps, grad_u, rho, u, grad_p,
                                       grad_k, two_invars, incl_p_invars, incl_tke_invars)  # ✓
    if incl_input_markers is True:
        x = np.concatenate((x, input_markers), axis=1)

    if incl_nut_input is True:
        nut = 0.09 * np.square(k) / eps
        r_nu = nut / ((100*5e-6) + nut)
        r_nu = np.expand_dims(r_nu, axis=1)
        x = np.concatenate((x, r_nu), axis=1)

    num_inputs = x.shape[1]
    tb = data_processor.calc_tensor_basis(Sij, Rij, num_tensor_basis)  # Tensor basis ✓
    y = data_processor.calc_true_output(tauij, output_var="bij")  # Output bij ✓
    print("x, tb and y calculations complete")

    # Enforce realizability
    if enforce_realiz is True:
        for i in range(num_realiz_its):
            y = data_processor.make_realizable(y)  # ✓
        print("Realizability enforced")

    return coords, x, tb, y, num_inputs


def set_seed(seed):  # ✓
    """
    Set random seeds for all operations with seed parameter. This ensures the results are
    reproducible.
    :param seed: integer between 1 and the value of maximum seed runs
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def split_database(coords, x, tb, y, train_list, valid_list, test_list, seed,
                   train_valid_rand_split, train_test_rand_split,
                   train_valid_split_frac, train_test_split_frac, case_dict):  # ✓
    """
    Split data using the random or specified data-splitting method. The random method
    randomly allocates (x, tb, y) data for TVT. The specified method allocates data
    from specific cases for TVT according to the TVT lists.
    TVT = training, validation and testing.

    :param coords: coordinates of data rows
    :param x: scalar inputs
    :param tb: tensor basis
    :param y: Reynolds stress anisotropy truth data
    :param train_list: Cases for training (use if train_test_rand_split = False)
    :param valid_list: Cases for validation (use if train_valid_rand_split = False)
    :param test_list: Cases for testing (use if train_test_rand_split = False)
    :param seed: current random seed for reproducible database splitting
    :param train_valid_rand_split: boolean for random train-valid splitting of training
           data
    :param train_test_rand_split: boolean for random train-test splitting of all data
    :param train_valid_split_frac: fraction of training data to use for actual training;
           the rest is used for validation (use if train_valid_rand_split = True)
    :param train_test_split_frac: fraction of all data to use for training and
           validation; the rest is used for testing (use if train_test_rand_split = True)
    :param case_dict: Dictionary specifying number of data points per case

    :return: coordinates, scalar inputs, tensor basis and outputs for the TVT data sets
    """

    # Split full database randomly into (train+validation) and test datasets if
    # train_test_rand_split = True ✓
    if train_test_rand_split is True:
        train_idx, test_idx = \
            DataSplitter.get_rand_split_idx(x, train_test_split_frac, seed)  # ✓
        coords_train, x_train, tb_train, y_train, coords_test, x_test, tb_test, y_test = \
            DataSplitter.tbnn_rand_split(coords, x, tb, y, train_idx, test_idx)  # ✓
        print("Train-test random data split complete")

        # Split (train+validation) dataset randomly into training and validation
        # datasets if train_valid_rand_split = True ✓
        if train_valid_rand_split is True:
            train_idx, valid_idx = \
                DataSplitter.get_rand_split_idx(x_train, train_valid_split_frac, seed)  # ✓
            coords_train, x_train, tb_train, y_train, coords_valid, x_valid, tb_valid, \
                y_valid = DataSplitter.tbnn_rand_split(coords_train, x_train, tb_train,
                                                       y_train, train_idx, valid_idx)  # ✓
            print("Train-valid random data split complete")
        else:
            coords_valid, x_valid, tb_valid, y_valid = [], [], [], []
            print("No data allocated for validation")

    # Else split full database according to specified cases if train_test_rand_split =
    # False ✓
    else:
        # Specified training database ✓
        idx = DataSplitter.get_spec_split_idx(train_list, case_dict, seed,
                                              shuffle=True)  # ✓
        coords_train, x_train, tb_train, y_train = \
            DataSplitter.tbnn_spec_split(coords, x, tb, y, idx)  # ✓
        print("Specified data split for training complete")

        # Specified testing database ✓
        idx = DataSplitter.get_spec_split_idx(test_list, case_dict, seed,
                                              shuffle=False)  # ✓
        coords_test, x_test, tb_test, y_test = \
            DataSplitter.tbnn_spec_split(coords, x, tb, y, idx)  # ✓
        print("Specified data split for testing complete")

        # Specified validation database ✓
        if train_valid_rand_split is False:
            idx = DataSplitter.get_spec_split_idx(valid_list, case_dict, seed,
                                                  shuffle=False)  # ✓
            coords_valid, x_valid, tb_valid, y_valid = \
                DataSplitter.tbnn_spec_split(coords, x, tb, y, idx)  # ✓
            print("Specified data split for validation complete")
        else:
            coords_valid, x_valid, tb_valid, y_valid = [], [], [], []
            print("No data allocated for validation")

    return coords_train, x_train, tb_train, y_train, coords_test, x_test, tb_test, \
        y_test, coords_valid, x_valid, tb_valid, y_valid


def tbnn_ops(x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test, tb_test,
             y_test, batch_size, num_hid_layers, num_hid_nodes, af, af_params, seed,
             weight_init, weight_init_params, loss, optimizer, init_lr, lr_scheduler,
             lr_scheduler_params, min_epochs, max_epochs, interval, avg_interval,
             print_freq, log, enforce_realiz, num_realiz_its, num_tensor_basis,
             num_inputs):  # ✓

    # Use GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare batch data in dataloaders and check dataloaders ✓
    dataloader = DataLoader(x_train, tb_train, y_train, x_valid, tb_valid, y_valid,
                            x_test, tb_test, y_test, batch_size)  # ✓
    DataLoader.check_data_loaders(dataloader.train_loader, x_train, tb_train, y_train,
                                  num_inputs, num_tensor_basis)  # ✓
    DataLoader.check_data_loaders(dataloader.valid_loader, x_valid, tb_valid, y_valid,
                                  num_inputs, num_tensor_basis)  # ✓
    DataLoader.check_data_loaders(dataloader.test_loader, x_test, tb_test, y_test,
                                  num_inputs, num_tensor_basis)  # ✓

    # Prepare TBNN architecture structure and check structure ✓
    structure = NetworkStructure(num_hid_layers, num_hid_nodes, af, af_params,
                                 num_inputs, num_tensor_basis)  # ✓
    structure.check_structure()  # ✓
    print("TBNN architecture structure and data loaders checked")

    # Construct TBNN and perform training, validation and testing ✓
    tbnn = Tbnn(device, seed, structure=structure, weight_init=weight_init,
                weight_init_params=weight_init_params).double()  # ✓
    tbnn_tvt = TbnnTVT(loss, optimizer, init_lr, lr_scheduler, lr_scheduler_params,
                       min_epochs, max_epochs, interval, avg_interval, print_freq, log,
                       tbnn)  # ✓
    epoch_count, final_train_rmse, final_valid_rmse = \
        tbnn_tvt.fit(device, dataloader.train_loader, dataloader.valid_loader, tbnn)  # ✓
    y_pred, test_rmse = tbnn_tvt.perform_test(device, enforce_realiz, num_realiz_its,
                                              log, dataloader.test_loader, tbnn)  # ✓

    return epoch_count, final_train_rmse, final_valid_rmse, y_pred, test_rmse


def trial_iter(num_seeds, coords, x, tb, y, train_list, valid_list, test_list,
               train_valid_rand_split, train_valid_split_frac, train_test_rand_split,
               train_test_split_frac, num_tensor_basis, num_hid_layers, num_hid_nodes,
               af, af_params, init_lr, lr_scheduler, lr_scheduler_params, weight_init,
               weight_init_params, max_epochs, min_epochs, interval, avg_interval, loss,
               optimizer, batch_size, enforce_realiz, num_realiz_its, folder_path,
               user_vars, print_freq, case_dict, num_inputs):  # ✓
    """
    After obtaining x, tb and y from preprocessing, run this function num_seeds times to
    train num_seeds TBNN instances and obtain num_seeds predictions:

    - Create new trial folder. A trial is a particular set-up of parameter values on
    the frontend.py file.
    - If seed = 1, initialise TVT error lists and write the parameter values.
    - Set the seed for all operations.
    - Split complete database into training, validation and test datasets.
    - Perform TBNN training, validation and testing using their respective datasets.
    - Write results for each TBNN instance (aka seed).
    - After all seeds for a particular trial, write results for the trial.
    """
    for seed in range(1, num_seeds+1):
        # Set up results logs, lists and files ✓
        current_folder, log = init_log(folder_path, seed)  # ✓
        if seed == 1:
            final_train_rmse_list, final_valid_rmse_list, test_rmse_list = \
                errors_list_init()  # ✓
            write_param_txt(current_folder, folder_path, user_vars)  # ✓
            write_param_csv(current_folder, folder_path, user_vars)  # ✓

        # Prepare TVT datasets ✓
        set_seed(seed)  # ✓
        coords_train, x_train, tb_train, y_train, coords_test, x_test, tb_test, y_test, \
            coords_valid, x_valid, tb_valid, y_valid = \
            split_database(coords, x, tb, y, train_list, valid_list, test_list, seed,
                           train_valid_rand_split, train_test_rand_split,
                           train_valid_split_frac, train_test_split_frac, case_dict)  # ✓
        if seed == 1:
            write_test_truth_bij(coords_test, folder_path, y_test)  # ✓

        # Run TBNN operations ✓
        epoch_count, final_train_rmse, final_valid_rmse, y_pred, test_rmse, = \
            tbnn_ops(x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test,
                     tb_test, y_test, batch_size, num_hid_layers, num_hid_nodes, af,
                     af_params, seed, weight_init, weight_init_params, loss, optimizer,
                     init_lr, lr_scheduler, lr_scheduler_params, min_epochs, max_epochs,
                     interval, avg_interval, print_freq, log, enforce_realiz,
                     num_realiz_its, num_tensor_basis, num_inputs)  # ✓

        # Write results for each seed ✓
        write_bij_results(coords_test, folder_path, seed, y_pred, current_folder)  # ✓
        final_train_rmse_list.append(final_train_rmse)
        final_valid_rmse_list.append(final_valid_rmse)
        test_rmse_list.append(test_rmse)

    # Write results for each trial ✓
    write_trial_rmse_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list,
                         folder_path, current_folder)  # ✓
    write_error_means_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list,
                          folder_path, current_folder)  # ✓

    return current_folder


def trial_iter_v2(num_seeds, x_train, tb_train, y_train, x_valid, tb_valid, y_valid,
                  coords_test, x_test, tb_test, y_test, num_tensor_basis, num_hid_layers,
                  num_hid_nodes, af, af_params, init_lr, lr_scheduler,
                  lr_scheduler_params, weight_init, weight_init_params, max_epochs,
                  min_epochs, interval, avg_interval, loss, optimizer, batch_size,
                  enforce_realiz, num_realiz_its, folder_path, user_vars, print_freq,
                  num_inputs):  # ✓
    """
    After obtaining x, tb and y from preprocessing, run this function num_seeds times to
    train num_seeds TBNN instances and obtain num_seeds predictions:

    - Create new trial folder. A trial is a particular set-up of parameter values on the
      frontend.py file.
    - If seed = 1, initialise TVT error lists and write the parameter values.
    - Set the seed for all operations.
    - Perform TBNN training, validation and testing.
    - Write results for each TBNN instance (aka seed).
    - After all seeds for a particular trial, write results for the trial.
    """
    for seed in range(1, num_seeds+1):
        # Set up results logs, lists and files ✓
        current_folder, log = init_log(folder_path, seed)  # ✓
        if seed == 1:
            final_train_rmse_list, final_valid_rmse_list, test_rmse_list = \
                errors_list_init()  # ✓
            write_param_txt(current_folder, folder_path, user_vars)  # ✓
            write_param_csv(current_folder, folder_path, user_vars)  # ✓
            write_test_truth_bij(coords_test, folder_path, y_test)  # ✓
        set_seed(seed)  # ✓

        # Run TBNN operations ✓
        epoch_count, final_train_rmse, final_valid_rmse, y_pred, test_rmse, = \
            tbnn_ops(x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test,
                     tb_test, y_test, batch_size, num_hid_layers, num_hid_nodes, af,
                     af_params, seed, weight_init, weight_init_params, loss, optimizer,
                     init_lr, lr_scheduler, lr_scheduler_params, min_epochs, max_epochs,
                     interval, avg_interval, print_freq, log, enforce_realiz,
                     num_realiz_its, num_tensor_basis, num_inputs)  # ✓

        # Write results for each seed ✓
        write_bij_results(coords_test, folder_path, seed, y_pred, current_folder)  # ✓
        final_train_rmse_list.append(final_train_rmse)
        final_valid_rmse_list.append(final_valid_rmse)
        test_rmse_list.append(test_rmse)

    # Write results for each trial ✓
    write_trial_rmse_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list,
                         folder_path, current_folder)  # ✓
    write_error_means_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list,
                          folder_path, current_folder)  # ✓

    return current_folder
