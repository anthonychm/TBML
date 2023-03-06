import os
import numpy as np
import pandas as pd
from pathlib import Path


def write_param_txt(current_folder, folder_path, user_vars):  # ✓
    # Write individual parameters.txt file in each trial folder
    var_name_list = ["trial", "num_hid_layers", "num_hid_nodes", "max_epochs",
                     "min_epochs", "interval", "avg_interval", "af", "af_params",
                     "weight_init", "weight_init_params", "init_lr", "lr_scheduler",
                     "lr_scheduler_params", "loss", "optimizer", "batch_size",
                     "incl_p_invars", "incl_tke_invars", "incl_input_markers",
                     "num_input_markers", "rho", "train_test_rand_split",
                     "train_test_split_frac", "train_valid_rand_split",
                     "train_valid_split_frac", "train_list", "valid_list", "test_list",
                     "num_dims"]

    with open(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder),
                           'Trial' + str(current_folder) + '_parameters.txt'), "a") as \
            vars_file:
        for var_name in var_name_list:
            if var_name == "trial":
                vars_file.write(str(var_name) + " " + str(current_folder) + "\n")
            else:
                vars_file.write(str(var_name) + " " + str(user_vars[var_name]) + "\n")


def write_param_csv(current_folder, folder_path, user_vars):  # ✓
    # Write new row in Trial_parameters_and_means.csv file ✓
    var_name_list = ["trial", "num_hid_layers", "num_hid_nodes", "max_epochs",
                     "min_epochs", "interval", "avg_interval", "af", "af_params",
                     "weight_init", "weight_init_params", "init_lr", "lr_scheduler",
                     "lr_scheduler_params", "loss", "optimizer", "batch_size",
                     "incl_p_invars", "incl_tke_invars", "incl_input_markers",
                     "num_input_markers", "rho", "train_test_rand_split",
                     "train_test_split_frac", "train_valid_rand_split",
                     "train_valid_split_frac", "train_list", "valid_list", "test_list",
                     "num_dims"]
    postprocess_name_list = ["run_time", "Mean_final_training_rmse",
                             "Mean_final_validation_rmse", "Mean_testing_rmse"]
    all_name_list = var_name_list + postprocess_name_list

    # Check if Trial_parameters_and_means.csv file exists ✓
    file_path = os.path.join(folder_path, "Results", "Trial_parameters_and_means.csv")
    path = Path(file_path)
    if path.is_file():
        pass
    else:
        # Create Trial_parameters_and_means.csv file
        main_params_df = pd.DataFrame(columns=all_name_list)
        main_params_df.to_csv(file_path, sep=",", index=False)

    # Read the csv file and add a new row of parameter values for current trial ✓
    main_params_df = pd.read_csv(file_path)
    for var_name in var_name_list:
        if var_name == "trial":
            main_params_df.loc[current_folder, var_name] = str(current_folder)
        else:
            main_params_df.loc[current_folder, var_name] = str(user_vars[var_name])
    main_params_df.to_csv(file_path, index=False)


def write_test_truth_logk(coords_test, folder_path, true_logk):  # ✓

    true_logk = np.hstack((coords_test, true_logk))
    np.savetxt(os.path.join(folder_path, 'LES_test_truth_logk.txt'), true_logk,
               delimiter=' ', header='Cx Cy log(k)')


def write_k_results(coords_test, folder_path, seed, predicted_logk, current_folder):  # ✓

    # Write log(k) and k results
    predicted_k = pow(10, predicted_logk)
    write_array = np.hstack((coords_test, predicted_logk, predicted_k))
    np.savetxt(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder),
                            'Trial' + str(current_folder) + '_seed' + str(seed) +
                            '_TKENN_test_prediction.txt'), write_array, delimiter=' ',
               header="Cx Cy log(k) k")
