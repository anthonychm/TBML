import numpy as np
import os.path
import pandas as pd


def write_all_mixture_results(coords_test, folder_path, seed, pi_all, mu_bij_all,
                               sigma_all, current_folder):
    # Write mixing coeffients to file
    pi_all = np.hstack((coords_test, pi_all))
    np.savetxt(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder),
                            'Trial' + str(current_folder) + '_seed' + str(seed) +
                            '_mixing_coefficients.txt'), pi_all, delimiter=' ',
               header="Cx Cy Mixing coefficients")

    # Write mean anisotropy to file
    for k in range(mu_bij_all.shape[0]):
        arr = np.hstack((coords_test, mu_bij_all[k, :, :]))
        np.savetxt(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder),
                                'Trial' + str(current_folder) + '_seed' + str(seed) +
                                '_mu_bij_from_kernel_' + str(k+1) + '.txt'), arr,
                   delimiter=' ', header="Cx Cy b11 b12 b13 b21 b22 b23 b31 b32 b33")

    # Write sigma to file
    sigma_all = np.hstack((coords_test, sigma_all))
    np.savetxt(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder),
                            'Trial' + str(current_folder) + '_seed' + str(seed) +
                            '_sigma.txt'), sigma_all, delimiter=' ', header="Cx Cy sigma")


def write_trial_loss_csv(final_train_avg_mnll_loss_list, final_valid_avg_mnll_loss_list,
                         test_avg_mnll_loss_list, test_rmse_list, folder_path,
                         current_folder):  #
    # Create list of random seeds used in current trial
    seed_list = [*range(len(final_train_avg_mnll_loss_list))]
    seed_list_plus = [seed + 1 for seed in seed_list]

    # Create pandas dataframe of loss and rmse lists
    trial_loss_csv = \
        pd.DataFrame(data={"Random seed": seed_list_plus,
                           "Final training loss": final_train_avg_mnll_loss_list,
                           "Final validation loss": final_valid_avg_mnll_loss_list,
                           "Testing loss": test_avg_mnll_loss_list,
                           "Testing rmse": test_rmse_list})

    # Specify order of columns in pandas dataframe
    trial_loss_csv = trial_loss_csv[["Random seed", "Final training loss",
                                     "Final validation loss", "Testing loss",
                                     "Testing rmse"]]
    # Write dataframe to csv file
    trial_loss_csv.to_csv(os.path.join(folder_path, 'Trials', 'Trial ' +
                                       str(current_folder), 'Trial' + str(current_folder)
                                       + '_loss_results.csv'), sep=',', index=False)


def write_loss_means_csv(final_train_avg_mnll_loss_list, final_valid_avg_mnll_loss_list,
                         test_avg_mnll_loss_list, test_rmse_list, folder_path,
                         current_folder):  #
    # Calculate means and store them in a dictionary
    means_dict = {"Mean_final_train_avg_mnll_loss": np.mean(final_train_avg_mnll_loss_list),
                  "Mean_final_valid_avg_mnll_loss": np.mean(final_valid_avg_mnll_loss_list),
                  "Mean_testing_avg_mnll_loss": np.mean(test_avg_mnll_loss_list),
                  "Mean_testing_rmse": np.mean(test_rmse_list)}

    # Write means in Trial_parameter_and_means.csv database
    main_params_df = pd.read_csv(os.path.join(folder_path, 'Results',
                                              'Trial_parameters_and_means.csv'))

    for mean in means_dict:
        main_params_df.loc[current_folder - 1, mean] = str(means_dict[mean])
    main_params_df.to_csv(os.path.join(folder_path, 'Results',
                                       'Trial_parameters_and_means.csv'), index=False)
