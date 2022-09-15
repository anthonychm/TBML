import numpy as np
import pandas as pd
from results_writer import load_channel_data
from results_writer import write_results
from results_writer import write_output
from results_writer import errors_list_init
from results_writer import write_error_means
from results_writer import write_trial_rmse_csv
from results_writer import write_test_truth_aij
from calculator import TurbulenceKEpsDataProcessor
import os
from core import NetworkStructure, TBNN # tbnn.py is now called core.py


def preprocessing(enforce_realizability, num_realizability_its, datafile_name, n_skiprows):
    """
    Preprocess the CFD data, which consists of the following steps:
    1) Load in data using load_channel_data function
    2) Calculate Sij and Rij for the RANS data using calc_Sij_Rij function
    3) Calculate invariants for the RANS data using calc_scalar_basis function
    4) Calculate tensor basis for the RANS data using calc_tensor_basis function
    5) Calculate Reynolds stress anisotropy for the LES/DNS/experimental data using calc_output function
    6) If enforce_realizability is True, enforce realizability in y using make_realizable function
    :param enforce_realizability: Boolean for enforcing realizability in y
    :param num_realizability_its: Number of iterations for enforcing realizability in y
    :param datafile_name: Name of the data file containing the CFD data
    :param n_skiprows: Number of rows at top of data file which this code should skip for reading
    :return: invariants x, tensor basis tb and Reynolds stress anisotropy y, for all of the CFD data
    """
    # Load in data
    k, eps, grad_u, stresses = load_channel_data(datafile_name = datafile_name, n_skiprows = n_skiprows)
    print('data loading complete')

    # Calculate inputs and outputs
    data_processor = TurbulenceKEpsDataProcessor()
    Sij, Rij = data_processor.calc_Sij_Rij(grad_u, k, eps) # Mean strain rate tensor Sij and mean rotation rate tensor Rij
    x = data_processor.calc_scalar_basis(Sij, Rij, is_train=True)  # Scalar basis
    tb = data_processor.calc_tensor_basis(Sij, Rij, quadratic_only=False)  # Tensor basis
    y = data_processor.calc_output(stresses)  # Anisotropy tensor
    print('x, tb and y calculations complete')

    # Enforce realizability
    if enforce_realizability is True:
        for i in range(num_realizability_its):
            y = TurbulenceKEpsDataProcessor.make_realizable(y)
        print('realisability enforced')

    return x, tb, y


def database_splitting(x, tb, y, seed, train_list, test_list, train_valid_random_split, valid_list,
                       train_test_random_split, train_test_split_fraction, n_case_points):
    """
    Split the data using your chosen data-splitting method
    :param x: scalar invariants
    :param tb: tensor basis
    :param y: Reynolds stress anisotropy
    :param seed: Current random seed for reproducible database splitting
    :param train_list: list of cases for training (use if random_split = False)
    :param test_list: list of cases for testing (use if random_split = False)
    :param train_test_random_split: Boolean for enforcing random train-test splitting of data (True) or specifying certain cases
    for training and testing (False)
    :param train_test_split_fraction: fraction of all CFD data to use for training data (use if random_split = True)
    :param n_case_points: Number of data points per CFD case
    :return: inputs, tb and outputs for the training and test sets
    """
    # Split all data into training and test datasets
    print("Random seed", seed)
    np.random.seed(seed)
    # ^ this sets the random seed for Theano when random_split is True
    # or sets random seed for specified split when random_split is False

    # .train_test_split and .train_test_specified are in preprocessor.py, called by TurbulenceKEpsDataProcessor
    if train_test_random_split is True:
        x_train, tb_train, y_train, x_test, tb_test, y_test = \
            TurbulenceKEpsDataProcessor.train_test_split(x, tb, y, fraction=train_test_split_fraction, seed=seed)
        x_valid, tb_valid, y_valid = [], [], []
    else:
        x_train, tb_train, y_train, x_test, tb_test, y_test, x_valid, tb_valid, y_valid = \
            TurbulenceKEpsDataProcessor.train_test_specified(x, tb, y, train_list, test_list, train_valid_random_split, valid_list,
                                                             n_case_points=n_case_points)

    return x_train, tb_train, y_train, x_test, tb_test, y_test, x_valid, tb_valid, y_valid


def tbnn_operations(num_layers, num_nodes, nonlinearity, nonlinearity_key, nonlinearity_key_value, train_valid_split_fraction,
                    learning_rate_decay, min_learning_rate, weight_initialiser_name, weight_initialiser_params,
                    x_train, tb_train, y_train, x_valid, tb_valid, y_valid, train_valid_random_split, max_epochs, min_epochs, init_learning_rate, interval, average_interval,
                    loss, optimizer, batch_size, x_test, tb_test, enforce_realizability, num_realizability_its, y_test,
                    output_file, seed, folder_path, current_folder, train_list, test_list, valid_list, train_test_random_split, train_test_split_fraction):
    """
    Conduct TBNN operations, which consists of the following steps:
    1) Set number of hidden layers and nodes and activation functions
    2) Build the TBNN architecture and fit the TBNN to the training data using the fit function
    3) With the TBNN, predict the Reynolds stress anisotropy in the training and test sets using the predict function
    4) If enforce realizability is True, make the TBNN predictions realizable using the make_realizable function
    5) Determine the errors in the TBNN predictions using the rmse_score function
    6) Write the parameter values used in each Trial folder and in Trial_parameters_and_means.csv database if seed == 1
    :param num_layers: Number of hidden layers
    :param num_nodes: Number of hidden nodes
    :param nonlinearity: Activation functions
    :param nonlinearity_key: Parameters of the activation function
    :param nonlinearity_key_value: Arguments of the activation function
    :param train_valid_split_fraction: Fraction of train data used for actual training after train-test splitting. The other
    fraction is used for validation.
    :param learning_rate_decay: Rate of decay for learning rate
    :param min_learning_rate: Minimum learning rate
    :param weight_initialiser_name: Weight initialiser algorithm
    :param weight_initialiser_params: Arguments of the weight initialiser algorithm
    :param x_train: scalar invariants in training set
    :param tb_train: tensor basis in training set
    :param y_train: Reynolds stress anisotropy values in training set
    :param max_epochs: Maximum number of training epochs
    :param min_epochs: Minimum number of training epochs
    :param init_learning_rate: Initial learning rate
    :param interval: Frequency at which convergence criteria are checked
    :param average_interval: Number of intervals averaged over to determine if early stopping criteria should be triggered
    :param loss: Loss function. If none specified, the default is a mean squared error loss function.
    :param optimizer: Optimizer algorithm
    :param batch_size: Number of data samples to use per batch
    :param x_test: scalar invariants in test set
    :param tb_test: tensor basis in test set
    :param enforce_realizability: Boolean for enforcing realizability in Reynolds stress anisotropy predictions
    :param num_realizability_its: Number of iterations for enforcing realizability in Reynolds stress anisotropy predictions
    :param y_test: Reynolds stress anisotropy values in test set
    :return: TBNN-predicted Reynolds stress anisotropy for the test set
    """
    # Define network structure
    structure = NetworkStructure()
    structure.set_num_layers(num_layers)
    structure.set_num_nodes(num_nodes)
    structure.set_nonlinearity(nonlinearity)
    structure.set_nonlinearity_keyword(nonlinearity_key, nonlinearity_key_value)

    # Initialize and fit TBNN
    tbnn = TBNN(structure, train_valid_split_fraction=train_valid_split_fraction, learning_rate_decay=learning_rate_decay,
                min_learning_rate=min_learning_rate)
    weight_initialiser = "lasagne.init." + weight_initialiser_name + "(" + weight_initialiser_params + ")"
    final_training_rmse, final_validation_rmse = tbnn.fit(x_train, tb_train, y_train, x_valid, tb_valid,
                                                                    y_valid, max_epochs=max_epochs, min_epochs=min_epochs,
                                                                    init_learning_rate=init_learning_rate, interval=interval,
                                                                    average_interval=average_interval, loss=loss, optimizer=optimizer,
                                                                    weight_initialiser=weight_initialiser, batch_size=batch_size, output_file=output_file,
                                                                    train_valid_random_split=train_valid_random_split)
    # Make predictions on train and test data to get train error and test error
    labels_train = tbnn.predict(x_train, tb_train)
    print('TBNN predict training aij complete')
    labels_test = tbnn.predict(x_test, tb_test)
    print('TBNN predict testing aij complete')

    # Re-enforce realizability
    if enforce_realizability:
        for i in range(num_realizability_its):
            labels_train = TurbulenceKEpsDataProcessor.make_realizable(labels_train)
            labels_test = TurbulenceKEpsDataProcessor.make_realizable(labels_test)

    # Determine error
    rmse_train = tbnn.rmse_score(y_train, labels_train)
    rmse_test = tbnn.rmse_score(y_test, labels_test)
    print("Deployment on training dataset rmse:", rmse_train)
    print("Deployment on test dataset rmse:", rmse_test)

    with open(output_file, "a") as write_file:
        print("Deployment on training dataset rmse:", rmse_train, file=write_file)
        print("Deployment on test dataset rmse:", rmse_test, file=write_file)

    deploy_on_training_set_rmse = rmse_train
    deploy_on_test_set_rmse = rmse_test

    # Write used parameters if seed == 1
    if seed == 1:
        excluded_vars = ["output_file", "i", "labels_test", "labels_train", "rmse_test", "rmse_train", "seed",
                         "structure", "tb_test", "tb_train", "tbnn", "vars_file", "weight_initialiser", "write_file",
                         "x_test", "x_train", "y_test", "y_train", "excluded_vars", "folder_path", "tb_valid", "x_valid",
                         "y_valid", "current_folder", "params_file", "main_params_df", "var_name", "writer",
                         "final_training_rmse", "final_validation_rmse", "deploy_on_training_set_rmse",
                         "deploy_on_test_set_rmse"]
        trial = current_folder
        # Write individual parameters file in each trial folder
        with open(os.path.join(folder_path, 'Trials', 'Trial ' + str(current_folder), 'Trial' + str(current_folder) +
                                            '_parameters.txt'), "a") as vars_file:
            for var_name in dir():
                if var_name in excluded_vars:
                    continue
                vars_file.write(str(var_name) + " " + str(eval(var_name)) + "\n")

        # Write new row in main parameters and means database
        main_params_df = pd.read_csv(os.path.join(folder_path, 'Results', 'Trial_parameters_and_means.csv'))
        for var_name in dir():
            if var_name in excluded_vars:
                continue
            main_params_df.loc[current_folder, var_name] = str(eval(var_name))
        main_params_df.to_csv(os.path.join(folder_path, 'Results', 'Trial_parameters_and_means.csv'), index=False)

    return labels_test, final_training_rmse, final_validation_rmse, deploy_on_training_set_rmse, deploy_on_test_set_rmse


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


def tbnn_iterate(seed_max, x, tb, y, train_list, test_list, train_valid_random_split, valid_list, train_test_random_split, train_test_split_fraction, n_case_points,
                 num_layers, num_nodes, nonlinearity, nonlinearity_key, nonlinearity_key_value, train_valid_split_fraction,
                 learning_rate_decay, min_learning_rate, weight_initialiser_name, weight_initialiser_params,
                 max_epochs, min_epochs, init_learning_rate, interval, average_interval, loss, optimizer, batch_size,
                 enforce_realizability, num_realizability_its, folder_path):

    for seed in range(1, seed_max):
        if seed == 1:
            final_training_rmse_list, final_validation_rmse_list, deploy_on_training_set_rmse_list, deploy_on_test_set_rmse_list = errors_list_init()
        current_folder, output_file = write_output(folder_path=folder_path, seed=seed)
        x_train, tb_train, y_train, x_test, tb_test, y_test, x_valid, tb_valid, y_valid = database_splitting(x=x, tb=tb, y=y, seed=seed,
                                                                                 train_list=train_list,
                                                                                 test_list=test_list,
                                                                                 train_valid_random_split=train_valid_random_split,
                                                                                 valid_list=valid_list,
                                                                                 train_test_random_split=train_test_random_split,
                                                                                 train_test_split_fraction=train_test_split_fraction,
                                                                                 n_case_points=n_case_points)
        write_test_truth_aij(folder_path, seed, true_aij=y_test)
        labels_test, final_training_rmse, final_validation_rmse, deploy_on_training_set_rmse, \
            deploy_on_test_set_rmse = tbnn_operations(num_layers=num_layers, num_nodes=num_nodes, nonlinearity=nonlinearity,
                                      nonlinearity_key=nonlinearity_key,
                                      nonlinearity_key_value=nonlinearity_key_value,
                                      train_valid_split_fraction=train_valid_split_fraction, learning_rate_decay=learning_rate_decay,
                                      min_learning_rate=min_learning_rate, weight_initialiser_name=weight_initialiser_name,
                                      weight_initialiser_params=weight_initialiser_params, x_train=x_train,
                                      tb_train=tb_train, y_train=y_train, x_valid=x_valid, tb_valid=tb_valid, y_valid=y_valid,
                                      train_valid_random_split=train_valid_random_split, max_epochs=max_epochs, min_epochs=min_epochs,
                                      init_learning_rate=init_learning_rate, interval=interval,
                                      average_interval=average_interval, loss=loss, optimizer=optimizer,
                                      batch_size=batch_size, x_test=x_test, tb_test=tb_test,
                                      enforce_realizability=enforce_realizability,
                                      num_realizability_its=num_realizability_its, y_test=y_test, seed=seed,
                                      folder_path=folder_path, current_folder=current_folder, train_list=train_list,
                                      test_list=test_list, valid_list=valid_list, output_file=output_file,
                                      train_test_random_split=train_test_random_split, train_test_split_fraction=train_test_split_fraction)
        write_results(folder_path=folder_path, seed=seed, predicted_aij=labels_test, current_folder=current_folder)
        del x_train, tb_train, y_train, x_test, tb_test, labels_test, y_test
        final_training_rmse_list.append(final_training_rmse)
        final_validation_rmse_list.append(final_validation_rmse)
        deploy_on_training_set_rmse_list.append(deploy_on_training_set_rmse)
        deploy_on_test_set_rmse_list.append(deploy_on_test_set_rmse)
    write_error_means(final_training_rmse_list, final_validation_rmse_list, deploy_on_training_set_rmse_list,
                      deploy_on_test_set_rmse_list, folder_path, current_folder)
    write_trial_rmse_csv(final_training_rmse_list, final_validation_rmse_list, deploy_on_training_set_rmse_list,
                         deploy_on_test_set_rmse_list, folder_path, current_folder)

    return current_folder


