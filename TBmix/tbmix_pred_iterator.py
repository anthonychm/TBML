import TBNN as tbnn
import tbmix_core
import tbmix_results_writer as tbmix_writer
import sys
sys.path.append('../TBNN')


def tbmix_ops(x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test, tb_test,
              y_test, batch_size, num_hid_layers, num_hid_nodes, af, af_params, seed,
              weight_init, weight_init_params, loss, optimizer, init_lr, lr_scheduler,
              lr_scheduler_params, min_epochs, max_epochs, interval, avg_interval,
              print_freq, log, enforce_realiz, num_realiz_its, num_tensor_basis,
              num_inputs, num_kernels, coords_test, test_list):

    # Prepare training batch data in data loader
    dl = tbnn.core.DataLoader(x_train, tb_train, y_train, x_valid, tb_valid, y_valid,
                            x_test, tb_test, y_test, batch_size)  # ✓
    tbnn.core.DataLoader.check_data_loaders(dl.train_loader, x_train, tb_train,
                                            y_train, num_inputs, num_tensor_basis)  # ✓
    tbnn.core.DataLoader.check_data_loaders(dl.valid_loader, x_valid, tb_valid,
                                            y_valid, num_inputs, num_tensor_basis)  # ✓

    # Prepare TBMix architecture structure and check structure
    structure = tbnn.core.NetworkStructure(num_hid_layers, num_hid_nodes, af,
                                           af_params, num_inputs, num_tensor_basis)  # ✓
    structure.check_structure()  # ✓
    print("TBMix architecture structure and data loaders checked")

    # Construct TBMix and perform training and validation
    tbmix = tbmix_core.TBMix(seed, num_kernels, structure=structure,
                             weight_init=weight_init,
                             weight_init_params=weight_init_params).double()  # ✓
    print(tbmix)
    tbmix_tvt = tbmix_core.TBMixTVT(optimizer, num_kernels, init_lr, lr_scheduler,
                                    lr_scheduler_params, min_epochs, max_epochs, interval,
                                    avg_interval, print_freq, log, tbmix)  # ✓
    epoch_count, train_loss_list, valid_loss_list = \
        tbmix_tvt.fit(dl.train_loader, dl.valid_loader, tbmix)  # ✓

    # Perform TBMix testing
    x_test, tb_test, y_test, pi_all, mu_bij_all, sigma_all, mu_bij_pred = \
        tbmix_tvt.test_preprocessing(x_test, tb_test, y_test)
    mu_bij_pred, avg_nll_loss, test_rmse, pi_all, mu_bij_all, sigma_all = \
        tbmix_tvt.perform_most_prob_test(x_test, tb_test, y_test, tbmix, pi_all,
                                         mu_bij_all, sigma_all, mu_bij_pred,
                                         enforce_realiz, num_realiz_its, log)
    # neigh_dict = tbmix_tvt.find_test_neighbours(test_list, coords_test)  # ✓
    # mu_bij_pred, avg_nll_loss, test_rmse, pi_all, mu_bij_all, sigma_all = \
    #     tbmix_tvt.perform_anchors_test(x_test, tb_test, y_test, tbmix, pi_all,
    #                                    mu_bij_all, sigma_all, mu_bij_pred, neigh_dict,
    #                                    enforce_realiz, num_realiz_its, log)  # ✓

    return epoch_count, train_loss_list, valid_loss_list, mu_bij_pred, avg_nll_loss, \
        test_rmse, pi_all, mu_bij_all, sigma_all


def trial_iter(num_seeds, coords, x, tb, y, train_list, valid_list, test_list,
               train_valid_rand_split, train_valid_split_frac, train_test_rand_split,
               train_test_split_frac, num_tensor_basis, num_hid_layers, num_hid_nodes,
               af, af_params, init_lr, lr_scheduler, lr_scheduler_params, weight_init,
               weight_init_params, max_epochs, min_epochs, interval, avg_interval, loss,
               optimizer, batch_size, enforce_realiz, num_realiz_its, folder_path,
               user_vars, print_freq, case_dict, num_inputs, num_kernels):  #

    """
    After obtaining x, tb and y from preprocessing, run this function num_seeds times to
    train num_seeds TBMix instances and obtain num_seeds predictions:

    - Create new trial folder. A trial is a particular set-up of parameter values in
    the tbmix_frontend.py file.
    - If seed = 1, initialise TVT error lists and write the parameter values.
    - Set the seed for all operations.
    - Split complete database into training, validation and test datasets.
    - Perform TBMix training, validation and testing using their respective datasets.
    - Write results for each TBMix instance (aka seed).
    - After all seed runs for a particular trial, write results for the trial.
    """

    for seed in range(1, num_seeds+1):
        # Set up results logs, lists and files
        current_folder, log = tbnn.write.init_log(folder_path, seed)  # ✓
        if seed == 1:
            final_train_avg_nll_loss_list = [None]*num_seeds
            final_valid_avg_nll_loss_list = [None]*num_seeds
            test_avg_nll_loss_list = [None]*num_seeds
            test_rmse_list = [None]*num_seeds
            tbnn.write.write_param_txt(current_folder, folder_path, user_vars) # ✓
            tbnn.write.write_param_csv(current_folder, folder_path, user_vars, model='tbmix')  # ✓

        # Prepare TVT datasets
        tbnn.piter.set_seed(seed)  # ✓
        coords_train, x_train, tb_train, y_train, coords_test, x_test, tb_test, y_test, \
            coords_valid, x_valid, tb_valid, y_valid = \
            tbnn.piter.split_database(coords, x, tb, y, train_list, valid_list, test_list,
                                      seed, train_valid_rand_split,
                                      train_test_rand_split, train_valid_split_frac,
                                      train_test_split_frac, case_dict)  # ✓
        if seed == 1:
            tbnn.write.write_test_truth_bij(coords_test, folder_path, y_test)  # ✓

        # Run TBMix operations
        epoch_count, train_loss_list, valid_loss_list, mu_bij_pred, avg_nll_loss, \
            test_rmse, pi_all, mu_bij_all, sigma_all = \
            tbmix_ops(x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test,
                      tb_test, y_test, batch_size, num_hid_layers, num_hid_nodes, af,
                      af_params, seed, weight_init, weight_init_params, loss,
                      optimizer, init_lr, lr_scheduler, lr_scheduler_params,
                      min_epochs, max_epochs, interval, avg_interval, print_freq, log,
                      enforce_realiz, num_realiz_its, num_tensor_basis, num_inputs,
                      num_kernels, coords_test, test_list)  # ✓

        # Write results for each seed
        tbnn.write.write_bij_results(coords_test, folder_path, seed, mu_bij_pred,
                                     current_folder)  # ✓
        tbmix_writer.write_all_mixture_results(coords_test, folder_path, seed, pi_all,
                                               mu_bij_all, sigma_all, current_folder)  # ✓
        final_train_avg_nll_loss_list[seed-1] = train_loss_list[-1]
        final_valid_avg_nll_loss_list[seed-1] = valid_loss_list[-1]
        test_avg_nll_loss_list[seed-1] = avg_nll_loss
        test_rmse_list[seed-1] = test_rmse

    # Write results for each trial
    tbmix_writer.write_trial_loss_csv(final_train_avg_nll_loss_list,
                                      final_valid_avg_nll_loss_list,
                                      test_avg_nll_loss_list, test_rmse_list,
                                      folder_path, current_folder)  # ✓
    tbmix_writer.write_loss_means_csv(final_train_avg_nll_loss_list,
                                      final_valid_avg_nll_loss_list,
                                      test_avg_nll_loss_list, test_rmse_list,
                                      folder_path, current_folder)  # ✓

    return current_folder
