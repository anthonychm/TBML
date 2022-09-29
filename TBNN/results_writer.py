import numpy as np
import os.path
import pandas as pd


def write_bij_results(folder_path, seed, predicted_bij, current_folder):
    """
    Write the bij values from testing to file.
    :param folder_path: Folder to save the results in
    :param seed: The current prediction's random seed
    :param predicted_bij: bij predicted by the TBNN
    """
    np.savetxt(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder), 'Trial' + str(current_folder) +
                            '_seed' + str(seed)+'_TBNN_test_prediction_bij.txt'), predicted_bij, delimiter=' ',
                            header=("uPrime2Mean uPrimevPrimeMean uPrimewPrimeMean vPrimeuPrimeMean vPrime2Mean "
                                    "vPrimewPrimeMean wPrimeuPrimeMean wPrimevPrimeMean wPrime2Mean"))


def write_test_truth_bij(folder_path, seed, true_bij):
    """
    Writes true bij values from LES.
    :param folder_path: Folder to save the results in
    :param seed: The current prediction's random seed
    :param true_bij: bij provided by LES
    """
    if seed == 1:
        np.savetxt(os.path.join(folder_path, 'LES_test_truth_bij.txt'), true_bij, delimiter = ' ',
                   header = ('uPrime2Mean uPrimevPrimeMean uPrimewPrimeMean vPrimeuPrimeMean vPrime2Mean '
                             'vPrimewPrimeMean wPrimeuPrimeMean wPrimevPrimeMean wPrime2Mean'))


def init_log(folder_path, seed):
    """
    Finds the highest Trial number and creates a new Trial folder if seed == 1. Creates new output log file for current
    seed.
    :param folder_path: Folder to save the results in
    :param seed: The current prediction's random seed
    :return: current folder: The trial index, output file: File path of output logs file for current seed
    """
    # Find highest trial folder number
    folder_counter = 0 # initialise folder counter
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
    log = os.path.join(folder_path, 'Trials', 'Trial '+str(current_folder), 'Trial'+str(current_folder)+'_seed'+str(seed)+'_output.txt')
    with open(log, "a") as write_file:
        print("Trial number =", current_folder, ", seed number =", seed, file=write_file)
    return current_folder, log


def write_time(start, stop, folder_path, current_folder):
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
    with open(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder), 'Trial' + str(current_folder) +
                                        '_parameters.txt'), "a") as vars_file:
        vars_file.write("run_time " + str(int(eval(run_time))) + "\n")

    # Write total run time in Trial_parameters_and_means.csv database
    main_params_df = pd.read_csv(os.path.join(folder_path, 'Results', 'Trial_parameters_and_means.csv'))
    main_params_df.loc[current_folder - 1, "run_time"] = str(eval(run_time))
    main_params_df.to_csv(os.path.join(folder_path, 'Results', 'Trial_parameters_and_means.csv'), index=False)


def errors_list_init():
    """Initialise RMSE lists as empty lists if seed == 1"""
    final_train_rmse_list, final_valid_rmse_list, test_rmse_list = [], [], []

    return final_train_rmse_list, final_valid_rmse_list, test_rmse_list


def write_param_txt(current_folder, folder_path, user_vars):
    # Write individual parameters.txt file in each trial folder
    var_incl_list = ["avg_interval", "batch_size", "enforce_realiz", "init_lr", "interval", "lr_decay", "loss",
                     "max_epochs", "min_epochs", "min_lr", "af", "af_key", "af_key_value", "num_layers", "num_nodes",
                     "num_realiz_its", "optimizer", "test_list", "train_list", "train_test_rand_split",
                     "train_test_split_frac", "train_valid_rand_split", "train_valid_split_frac", "trial",
                     "valid_list", "weight_init_name", "weight_init_params"]
    trial = current_folder
    with open(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder), 'Trial' + str(current_folder) +
                                                                                  '_parameters.txt'), "a") as vars_file:
        for var_name in var_incl_list:
            if var_name == "trial":
                vars_file.write(str(var_name) + " " + str(eval(var_name)) + "\n")
            vars_file.write(str(var_name) + " " + str(eval(user_vars[var_name])) + "\n")


def write_param_csv(current_folder, folder_path, user_vars):
    # Write new row in parameters and means database
    var_incl_list = ["avg_interval", "batch_size", "enforce_realiz", "init_lr", "interval", "lr_decay", "loss",
                     "max_epochs", "min_epochs", "min_lr", "af", "af_key", "af_key_value", "num_layers", "num_nodes",
                     "num_realiz_its", "optimizer", "test_list", "train_list", "train_test_rand_split",
                     "train_test_split_frac", "train_valid_rand_split", "train_valid_split_frac", "trial",
                     "valid_list", "weight_init_name", "weight_init_params"]
    trial = current_folder
    main_params_df = pd.read_csv(os.path.join(folder_path, 'Results', 'Trial_parameters_and_means.csv'))
    for var_name in var_incl_list:
        if var_name == "trial":
            main_params_df.loc[current_folder, var_name] = str(eval(var_name))
        main_params_df.loc[current_folder, var_name] = str(eval(user_vars[var_name]))
    main_params_df.to_csv(os.path.join(folder_path, 'Results', 'Trial_parameters_and_means.csv'), index=False)


def write_error_means_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list, folder_path, current_folder):
    """Write mean rmse statistics for each trial in Trial_parameters_and_means.csv database"""
    # Calculate means and store them in a dictionary
    means_dict = {
        "Mean_final_training_rmse": np.mean(final_train_rmse_list),
        "Mean_final_validation_rmse": np.mean(final_valid_rmse_list),
        "Mean_testing_rmse": np.mean(test_rmse_list)
    }
    # Write means in Trial_parameter_and_means.csv database
    main_params_df = pd.read_csv(os.path.join(folder_path, 'Results', 'Trial_parameters_and_means.csv'))

    for mean in means_dict:
        main_params_df.loc[current_folder - 1, mean] = str(means_dict[mean])
    main_params_df.to_csv(os.path.join(folder_path, 'Results', 'Trial_parameters_and_means.csv'), index=False)


def write_trial_rmse_csv(final_train_rmse_list, final_valid_rmse_list, test_rmse_list, folder_path, current_folder):
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
    trial_rmse_csv = trial_rmse_csv[["Random seed", "Final training rmse", "Final validation rmse", "Testing rmse"]]
    # Write dataframe to csv file
    trial_rmse_csv.to_csv(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder), 'Trial' +
                                  str(current_folder) + '_rmse_results.csv'), sep=',', index=False)
