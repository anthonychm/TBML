import numpy as np
import os.path
import pandas as pd
from pathlib import Path


def create_parent_folders():  # ✓

    # Create TBNN output data folder if it does not exist
    folder_path = os.path.join(os.getcwd(), "TBNN output data")
    path = Path(folder_path)
    if path.is_dir():
        pass
    else:
        os.mkdir(folder_path)

    # Create Results and Trials folders if they do not exist
    results_path = os.path.join(folder_path, "Results")
    trials_path = os.path.join(folder_path, "Trials")
    for path in [results_path, trials_path]:
        path_instance = Path(path)
        if path_instance.is_dir():
            pass
        else:
            os.mkdir(path)

    return folder_path


def write_bij_results(coords_test, folder_path, seed, predicted_bij, current_folder):  # ✓
    """
    Write bij results to file.
    :param coords_test: Coordinates of test points
    :param folder_path: The parent folder path
    :param seed: The current prediction's random seed
    :param predicted_bij: bij predicted by the TBNN
    :param current_folder: Current trial number
    """

    predicted_bij = np.hstack((coords_test, predicted_bij))
    np.savetxt(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder),
                            'Trial' + str(current_folder) + '_seed' + str(seed) +
                            '_TBNN_test_prediction_bij.txt'), predicted_bij,
               delimiter=' ', header="Cx Cy b11 b12 b13 b21 b22 b23 b31 b32 b33")


def write_test_truth_bij(coords, folder_path, true_bij):  # ✓
    """
    Writes true bij values from LES/DNS/experiment.
    :param coords: Coordinates of data rows
    :param folder_path: Folder to save the results in
    :param true_bij: bij provided by LES/DNS/experiment
    """

    true_bij = np.hstack((coords, true_bij))
    np.savetxt(os.path.join(folder_path, 'LES_test_truth_bij.txt'), true_bij,
               delimiter=' ', header='Cx Cy b11 b12 b13 b21 b22 b23 b31 b32 b33')


def init_log(folder_path, seed):  # ✓
    """
    Finds the highest Trial number. If seed == 1, creates a new Trial folder as current
    trial folder, else continue using current trial folder. Creates new output log file
    for new seed in current trial folder.
    :param folder_path: Working directory path for saving the results in
    :param seed: The current prediction's random seed
    :return: current folder: The trial index, log: File path of output log file for new
             seed
    """
    # Find highest trial folder number
    folder_counter = 0  # initialise folder counter
    for folder_name in os.listdir(os.path.join(folder_path, 'Trials')):
        for character in folder_name:
            if character.isspace():
                folder_counter = folder_counter + 1

    # Make new trial folder if seed == 1
    if seed == 1:
        new_folder = folder_counter + 1
        os.mkdir(os.path.join(folder_path, 'Trials', 'Trial '+str(new_folder)))
        current_folder = new_folder
    # Else keep current trial folder number
    else:
        current_folder = folder_counter

    print("Trial number =", current_folder, ", seed number =", seed)
    # Create new output log file in current trial folder
    log = os.path.join(folder_path, 'Trials', 'Trial '+str(current_folder),
                       'Trial'+str(current_folder)+'_seed'+str(seed)+'_output.txt')
    with open(log, "a") as write_file:
        print("Trial number =", current_folder, ", seed number =", seed, file=write_file)

    return current_folder, log


def write_time(start, stop, folder_path, current_folder):  # ✓
    """
    Calculates total run time for all random seeds for current trial
    :param start: timeit.default_timer() - see frontend.py
    :param stop: timeit.default_timer() - see frontend.py
    :param folder_path: Folder to save the results in
    :param current_folder: The trial index
    """
    # Calculate total run time
    run_time = str(stop - start)
    # Append total run time in .txt parameters file
    with open(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder),
                           'Trial' + str(current_folder) + '_parameters.txt'), "a") as \
            vars_file:
        vars_file.write("run_time " + str(float(run_time)) + "\n")

    # Write total run time in Trial_parameters_and_means.csv database
    main_params_df = pd.read_csv(os.path.join(folder_path, 'Results',
                                              'Trial_parameters_and_means.csv'))
    main_params_df.loc[current_folder - 1, "run_time"] = str(float(run_time))
    main_params_df.to_csv(os.path.join(folder_path, 'Results',
                                       'Trial_parameters_and_means.csv'), index=False)


def errors_list_init():
    """Initialise RMSE lists as empty lists if seed == 1"""
    final_train_rmse_list, final_valid_rmse_list, test_rmse_list = [], [], []

    return final_train_rmse_list, final_valid_rmse_list, test_rmse_list


def write_param_txt(current_folder, folder_path, user_vars):  # ✓
    # Write parameters.txt file in each trial folder
    var_name_list = ["trial", "num_hid_layers", "num_hid_nodes", "num_tensor_basis",
                     "max_epochs", "min_epochs", "interval", "avg_interval",
                     "enforce_realiz", "num_realiz_its", "af", "af_params",
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
    # Write new row in Trial_parameters_and_means.csv file
    var_name_list = ["trial", "num_hid_layers", "num_hid_nodes", "num_tensor_basis",
                     "max_epochs", "min_epochs", "interval", "avg_interval",
                     "enforce_realiz", "num_realiz_its", "af", "af_params",
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

    # Check if Trial_parameters_and_means.csv file exists
    file_path = os.path.join(folder_path, "Results", "Trial_parameters_and_means.csv")
    path = Path(file_path)
    if path.is_file():
        pass
    else:
        # Create Trial_parameters_and_means.csv file
        main_params_df = pd.DataFrame(columns=all_name_list)
        main_params_df.to_csv(file_path, sep=",", index=False)

    # Read the csv file and add a new row of parameter values for current trial
    main_params_df = pd.read_csv(file_path)
    for var_name in var_name_list:
        if var_name == "trial":
            main_params_df.loc[current_folder, var_name] = str(current_folder)
        else:
            main_params_df.loc[current_folder, var_name] = str(user_vars[var_name])
    main_params_df.to_csv(file_path, index=False)


def write_error_means_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list,
                          folder_path, current_folder):  # ✓
    """Write mean RMSEs for each trial in Trial_parameters_and_means.csv database"""
    # Calculate means and store them in a dictionary
    means_dict = {
        "Mean_final_training_rmse": np.mean(final_train_rmse_list),
        "Mean_final_validation_rmse": np.mean(final_valid_rmse_list),
        "Mean_testing_rmse": np.mean(test_rmse_list)
    }
    # Write means in Trial_parameter_and_means.csv database
    main_params_df = pd.read_csv(os.path.join(folder_path, 'Results',
                                              'Trial_parameters_and_means.csv'))

    for mean in means_dict:
        main_params_df.loc[current_folder - 1, mean] = str(means_dict[mean])
    main_params_df.to_csv(os.path.join(folder_path, 'Results',
                                       'Trial_parameters_and_means.csv'), index=False)


def write_trial_rmse_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list,
                         folder_path, current_folder):  # ✓
    """Write all rmse results in a csv file for each trial"""
    # Create list of random seeds used in current trial
    seed_list = [*range(len(final_train_rmse_list))]
    seed_list_plus = [seed + 1 for seed in seed_list]
    # Create pandas dataframe of RMSE lists
    trial_rmse_csv = pd.DataFrame(data={"Random seed": seed_list_plus,
                                        "Final training rmse": final_train_rmse_list,
                                        "Final validation rmse": final_valid_rmse_list,
                                        "Testing rmse": test_rmse_list})
    # Specify order of columns in pandas dataframe
    trial_rmse_csv = trial_rmse_csv[["Random seed", "Final training rmse",
                                     "Final validation rmse", "Testing rmse"]]
    # Write dataframe to csv file
    trial_rmse_csv.to_csv(os.path.join(folder_path, 'Trials', 'Trial ' +
                                       str(current_folder), 'Trial' + str(current_folder)
                                       + '_rmse_results.csv'), sep=',', index=False)
