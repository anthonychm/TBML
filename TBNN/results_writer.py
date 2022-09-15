import numpy as np
import matplotlib.pyplot as plt
import os.path
import pandas as pd


def load_channel_data(datafile_name, n_skiprows):
    """
    Load CFD data for TBNN calculations.
    :param datafile_name: Name of the data file containing the CFD data. The columns must be in the following order: TKE
    from RANS, epsilon from RANS, velocity gradients from RANS and Reynolds stresses from high-fidelity simulation.
    :param n_skiprows: Number of rows at top of data file which this code should skip for reading
    :return: Variables k and eps from RANS, flattened grad_u tensor from RANS and flattened Reynolds stress tensor from
    high-fidelity simulation.
    """

    # Load in F-BFS data
    data = np.loadtxt(datafile_name, skiprows = n_skiprows)
    k = data[:, 0]
    eps = data[:, 1]
    grad_u_flat = data[:, 2:11]
    stresses_flat = data[:, 11:]

    # Reshape grad_u and stresses to num_points X 3 X 3 arrays
    num_points = data.shape[0]
    grad_u = np.zeros((num_points, 3, 3))
    stresses = np.zeros((num_points, 3, 3))
    for i in range(3):
        for j in range(3):
            grad_u[:, i, j] = grad_u_flat[:, i*3+j]
            stresses[:, i, j] = stresses_flat[:, i*3+j]
    return k, eps, grad_u, stresses


def plot_results(predicted_stresses, true_stresses):
    """
    Create a plot with 9 subplots.  Each subplot shows the predicted vs the true value of that
    stress anisotropy component.  Correct predictions should lie on the y=x line (shown with
    red dash).
    :param predicted_stresses: Predicted Reynolds stress anisotropy (from TBNN predictions)
    :param true_stresses: True Reynolds stress anisotropy (from high fidelity simulations)
    """
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    on_diag = [0, 4, 8]
    for i in range(9):
            plt.subplot(3, 3, i+1)
            ax = fig.gca()
            ax.set_aspect('equal')
            plt.plot([-1., 1.], [-1., 1.], 'r--')
            plt.scatter(true_stresses[:, i], predicted_stresses[:, i], marker = 'x')
            plt.xlabel('True value')
            plt.ylabel('Predicted value')

            # Row index
            if i+1 < 4:
                idx_1 = 1
            elif 4 <= i+1 < 7:
                idx_1 = 2
            elif 7 <= i+1 < 10:
                idx_1 = 3
            else:
                idx_1 = 'undefined'

            # Column index
            if i+1 == 1 or i+1 == 4 or i+1 == 7:
                idx_2 = 1
            elif i+1 == 2 or i+1 == 5 or i+1 == 8:
                idx_2 = 2
            elif i+1 == 3 or i+1 == 6 or i+1 == 9:
                idx_2 = 3
            else:
                idx_2 = 'undefined'

            plt.title('a' + str(idx_1) + str(idx_2))

            # if i in on_diag:
            #     plt.xlim([-1./3., 2./3.])
            #     plt.ylim([-1./3., 2./3.])
            # else:
            # plt.xlim([-0.5, 0.5])
            # plt.ylim([-0.5, 0.5])
    plt.tight_layout()
    plt.show()
    print('plot complete')


def write_results(folder_path, seed, predicted_aij, current_folder):
    """
    Write the aij values from testing to file.
    :param folder_path: Folder to save the results in
    :param seed: The current prediction's random seed
    :param predicted_aij: aij predicted by the TBNN
    """
    np.savetxt(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder), 'Trial' + str(current_folder) +
                            '_seed' + str(seed)+'_TBNN_test_prediction_aij.txt'), predicted_aij, delimiter = ' ',
                            header = ('uPrime2Mean uPrimevPrimeMean uPrimewPrimeMean vPrimeuPrimeMean vPrime2Mean '
                            'vPrimewPrimeMean wPrimeuPrimeMean wPrimevPrimeMean wPrime2Mean'))


def write_test_truth_aij(folder_path, seed, true_aij):
    """
    Writes true aij values from LES.
    :param folder_path: Folder to save the results in
    :param seed: The current prediction's random seed
    :param true_aij: aij provided by LES
    """
    if seed == 1:
        np.savetxt(os.path.join(folder_path, 'LES_test_truth_aij.txt'), true_aij, delimiter = ' ',
                   header = ('uPrime2Mean uPrimevPrimeMean uPrimewPrimeMean vPrimeuPrimeMean vPrime2Mean '
                             'vPrimewPrimeMean wPrimeuPrimeMean wPrimevPrimeMean wPrime2Mean'))


def write_output(folder_path, seed):
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
    output_file = os.path.join(folder_path, 'Trials', 'Trial '+str(current_folder), 'Trial'+str(current_folder)+'_seed'+str(seed)+'_output.txt')
    with open(output_file, "a") as write_file:
        print("Trial number =", current_folder, ", seed number =", seed, file=write_file)
    return current_folder, output_file


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
    # Append total run time in parameters file
    with open(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder), 'Trial' + str(current_folder) +
                                        '_parameters.txt'), "a") as vars_file:
        vars_file.write("run_time " + str(int(eval(run_time))) + "\n")

    # Write total run time in Trial_parameters_and_means.csv database
    main_params_df = pd.read_csv(os.path.join(folder_path, 'Results', 'Trial_parameters_and_means.csv'))
    main_params_df.loc[current_folder - 1, "run_time"] = str(eval(run_time))
    main_params_df.to_csv(os.path.join(folder_path, 'Results', 'Trial_parameters_and_means.csv'), index=False)


def errors_list_init():
    """Initialise RMSE lists as empty lists if seed == 1"""
    final_training_rmse_list = []
    final_validation_rmse_list = []
    deploy_on_training_set_rmse_list = []
    deploy_on_test_set_rmse_list = []

    return final_training_rmse_list, final_validation_rmse_list, deploy_on_training_set_rmse_list, deploy_on_test_set_rmse_list


def write_error_means(final_training_rmse_list, final_validation_rmse_list, deploy_on_training_set_rmse_list,
                      deploy_on_test_set_rmse_list, folder_path, current_folder):
    """Write mean rmse statistics in Trial_parameters_and_means.csv database"""
    # Calculate means and store them in a dictionary
    means_dict = {
        "mean_final_training_rmse": np.mean(final_training_rmse_list),
        "mean_final_validation_rmse": np.mean(final_validation_rmse_list),
        "mean_deploy_on_training_set_rmse": np.mean(deploy_on_training_set_rmse_list),
        "mean_deploy_on_test_set_rmse": np.mean(deploy_on_test_set_rmse_list)
    }
    # Write means in Trial_parameter_and_means.csv database
    main_params_df = pd.read_csv(os.path.join(folder_path, 'Results', 'Trial_parameters_and_means.csv'))

    for mean in means_dict:
        main_params_df.loc[current_folder - 1, mean] = str(means_dict[mean])
    main_params_df.to_csv(os.path.join(folder_path, 'Results', 'Trial_parameters_and_means.csv'), index=False)


def write_trial_rmse_csv(final_training_rmse_list, final_validation_rmse_list, deploy_on_training_set_rmse_list,
                         deploy_on_test_set_rmse_list, folder_path, current_folder):
    """Write all rmse results in a csv file for each trial"""
    # Create list of random seeds used in current trial
    seed_list = [*range(len(final_training_rmse_list))]
    seed_list_plus = [seed + 1 for seed in seed_list]
    # Create pandas dataframe of RMSE lists
    trial_rmse_csv = pd.DataFrame(data={"Random seed": seed_list_plus,
                                   "Final training rmse": final_training_rmse_list,
                                   "Final validation rmse": final_validation_rmse_list,
                                   "Deploy on training set rmse": deploy_on_training_set_rmse_list,
                                   "Deploy on test set rmse": deploy_on_test_set_rmse_list})
    # Specify order of columns in pandas dataframe
    trial_rmse_csv = trial_rmse_csv[["Random seed", "Final training rmse", "Final validation rmse",
                                     "Deploy on training set rmse", "Deploy on test set rmse"]]
    # Write dataframe to csv file
    trial_rmse_csv.to_csv(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder), 'Trial' +
                                  str(current_folder) + '_rmse_results.csv'), sep=',', index=False)
