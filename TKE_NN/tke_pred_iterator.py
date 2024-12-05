import numpy as np
import TBNN as tbnn
from tke_core import tkenn_ops
from tke_preprocessor import calc_output, split_database
from tke_results_writer import write_param_txt, write_param_csv, write_test_truth_logk, \
    write_k_results, write_k_results_v2


def preprocessing(database, num_dims, num_input_markers, num_zonal_markers, two_invars,
                  incl_p_invars, incl_tke_invars, incl_input_markers, incl_zonal_markers,
                  rho, incl_rans_k):
    # Load in data ✓
    coords, k, eps, grad_u, grad_p, u, grad_k, input_markers, tauij = \
        tbnn.preprocessor.load_data(database, num_dims, num_input_markers,
                               num_zonal_markers, incl_p_invars, incl_tke_invars,
                               incl_input_markers, incl_zonal_markers)  # ✓
    print("Data loading complete")

    # Calculate inputs and outputs ✓
    data_processor = tbnn.calculator.PopeDataProcessor()  # ✓
    _, _, x = tbnn.preprocessor.scalar_basis_manager(data_processor, k, eps, grad_u, rho,
                                                u, grad_p, grad_k, two_invars,
                                                incl_p_invars, incl_tke_invars)  # ✓
    if incl_input_markers is True:
        x = np.concatenate((x, input_markers), axis=1)  # ✓
    if incl_rans_k is True:
        # Log-normalize k
        x = np.concatenate((x, np.log10(np.expand_dims(k, axis=1))), axis=1)
    num_inputs = x.shape[1]
    y = calc_output(tauij)  # y = log(k) ✓
    print("x and y calculations complete")

    return coords, x, y, num_inputs


def trial_iter(num_seeds, coords, x, y, train_list, valid_list, test_list,
               train_valid_rand_split, train_valid_split_frac, train_test_rand_split,
               train_test_split_frac, num_hid_layers, num_hid_nodes, af, af_params,
               init_lr, lr_scheduler, lr_scheduler_params, weight_init,
               weight_init_params, max_epochs, min_epochs, interval, avg_interval, loss,
               optimizer, batch_size, folder_path, user_vars, print_freq, case_dict,
               num_inputs):

    # Loop the following for each instance (seed) of TKENN:
    for seed in range(1, num_seeds + 1):
        # Set up results logs, lists and files ✓
        current_folder, log = tbnn.results_writer.init_log(folder_path, seed)  # ✓
        if seed == 1:
            final_train_rmse_list, final_valid_rmse_list, test_rmse_list = \
                tbnn.results_writer.errors_list_init()  # ✓
            write_param_txt(current_folder, folder_path, user_vars)  # ✓
            write_param_csv(current_folder, folder_path, user_vars)  # ✓

        # Prepare TVT datasets ✓
        tbnn.pred_iterator.set_seed(seed)  # ✓
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
            tkenn_ops(x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size,
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
    tbnn.results_writer.write_trial_rmse_csv(final_train_rmse_list, final_valid_rmse_list,
                                        test_rmse_list, folder_path,
                                        current_folder)  # ✓
    tbnn.results_writer.write_error_means_csv(final_train_rmse_list,
                                              final_valid_rmse_list,
                                         test_rmse_list, folder_path,
                                         current_folder)  # ✓

    return current_folder


def trial_iter_v2(num_seeds, x_train, y_train, x_valid, y_valid, coords_test, x_test,
                  y_test, num_hid_layers, num_hid_nodes, af, af_params, init_lr,
                  lr_scheduler, lr_scheduler_params, weight_init, weight_init_params,
                  max_epochs, min_epochs, interval, avg_interval, loss, optimizer,
                  batch_size, folder_path, user_vars, print_freq, num_inputs,
                  test_case_tags, test_output_normzr, zone):

    # Loop the following for each instance (seed) of TKENN:
    for seed in range(1, num_seeds + 1):
        # Set up results logs, lists and files ✓
        current_folder, log = tbnn.results_writer.init_log(folder_path, seed)  # ✓
        if seed == 1:
            final_train_rmse_list, final_valid_rmse_list, test_rmse_list = \
                tbnn.results_writer.errors_list_init()  # ✓
            write_param_txt(current_folder, folder_path, user_vars)  # ✓
            write_param_csv(current_folder, folder_path, user_vars)  # ✓
            write_test_truth_logk(coords_test, folder_path, y_test)  # ✓
        tbnn.pred_iterator.set_seed(seed)  # ✓

        # Run TKENN operations ✓
        epoch_count, final_train_rmse, final_valid_rmse, y_pred, test_rmse = \
            tkenn_ops(x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size,
                      num_hid_layers, num_hid_nodes, af, af_params, seed,
                      weight_init, weight_init_params, loss, optimizer, init_lr,
                      lr_scheduler, lr_scheduler_params, min_epochs, max_epochs,
                      interval, avg_interval, print_freq, log, num_inputs)  # ✓

        # Write results for each seed ✓
        write_k_results_v2(coords_test, folder_path, seed, y_pred, current_folder,
                           test_case_tags, test_output_normzr, zone)  # ✓
        final_train_rmse_list.append(final_train_rmse)
        final_valid_rmse_list.append(final_valid_rmse)
        test_rmse_list.append(test_rmse)

    # Write results for each trial ✓
    tbnn.results_writer.write_trial_rmse_csv(final_train_rmse_list, final_valid_rmse_list,
                                        test_rmse_list, folder_path,
                                        current_folder)  # ✓
    tbnn.results_writer.write_error_means_csv(final_train_rmse_list,
                                              final_valid_rmse_list,
                                         test_rmse_list, folder_path,
                                         current_folder)  # ✓

    return current_folder
