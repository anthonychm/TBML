###############################################################
#
# Copyright 2017 Sandia Corporation. Under the terms of
# Contract DE-AC04-94AL85000 with Sandia Corporation, the
# U.S. Government retains certain rights in this software.
# This software is distributed under the BSD-3-Clause license.
#
##############################################################

# This Python file was made by Anthony Man, University of Manchester.
# Run TBNN by executing this script.

"""
In this example, a Tensor Basis Neural Network (TBNN) is trained on the data for F-BFS flows.
The inputs are the mean strain rate tensor Sij and the mean rotation rate tensor Rij (normalized by k and epsilon).
The output is the Reynolds stress anisotropy tensor aij.

Data:
The F-BFS data set is based on 2D data from LES of B-BFS with inlet velocity = 1, 2 and 4 m/s.
The tke, epsilon, and velocity gradients are from k-omega SST RANS simulations for the same flows.

Reference for data driven turbulence modeling TBNN implementation:
Ling, J., Kurzawski, A. and Templeton, J., 2016. Reynolds averaged turbulence modelling using deep neural
networks with embedded invariance. Journal of Fluid Mechanics, 807, pp.155-166.
"""
import os

from pred_iterator import preprocessing
from pred_iterator import tbnn_iterate
import timeit
from results_writer import write_time


def main():
    # Define parameters
    num_layers = 5  # Number of hidden layers in the TBNN, default = 2
    num_nodes = 50  # Number of nodes per hidden layer, default = 10
    max_epochs = 100000  # Max number of epochs during training, default = 2000
    min_epochs = 60  # Min number of training epochs required, default = 1000
    interval = 10  # Frequency at which convergence is checked, default = 100
    average_interval = 3  # Number of intervals averaged over for early stopping criteria, default = 4
    enforce_realizability = True  # Boolean for enforcing realizability on Reynolds stresses, default = True
    num_realizability_its = 5  # Number of iterations for realizability enforcing, default = 5

    # Define advanced parameters
    nonlinearity = "elu" # Nonlinear activation function used, default = "LeakyRectify"
    nonlinearity_key = "" # Parameters of the nonlinear activation function, default = "leakiness"
    nonlinearity_key_value = "" # Arguments to the parameters of the nonlinear activation function, default = "0.1"
    weight_initialiser_name = "HeUniform" # Weight initialiser algorithm used, default = "HeUniform"
    weight_initialiser_params = "gain=np.sqrt(2.0)" # Arguments of the weight initialiser algorithm, default = "gain=np.sqrt(2.0)"
    init_learning_rate = 0.001 # Initial learning rate, default = 0.01
    min_learning_rate = 1e-6 # Minimum learning rate, default = 1e-6
    learning_rate_decay = 1 # Rate of decay for learning rate, default = 1
    loss_function = None # Loss function for comparing predictions and targets, default = None (None uses mean squared error)
    optimizer = "adam" # Optimizer algorithm, default = "adam"
    batch_size = 5 # Batch size for training, default = 1

    # Define data and how it is split for training and testing
    datafile_name = 'FBFS_full_set_no_walls.txt'
    n_skiprows = 1 # Skip this first number of rows in datafile_name for reading
    train_test_random_split = False
    train_test_split_fraction = 0  # Fraction of data allocated for training if using random split method, default = 0.8

    train_valid_random_split = False # Take some % of data at the end of train_list for validation, default = True
    train_valid_split_fraction = 0 # After running train-test split, this is the fraction of train data used for actual training. The other fraction is used for validation. Default = 0.9
    # Available inlet velocities [1, 2, 4]
    train_list = [1]
    valid_list = [4] # If train_valid_random_split = True, this parameter should be []
    test_list = [2]
    n_case_points = 80296 # Number of data points in each CFD case
    n_seeds = 5 # Number of reproducible TBNN predictions to save
    seed_max = n_seeds + 1

    # Write folder
    main_dir = os.getcwd()
    folder_path = os.path.join(main_dir, 'TBNN output data')

    """
    This script runs two main functions: preprocessing and tbnn_iterate. These are functions of other functions. See 
    prediction_iterator.py for more information on them.
    """
    start = timeit.default_timer()
    x, tb, y = preprocessing(enforce_realizability=enforce_realizability,
                             num_realizability_its=num_realizability_its, datafile_name=datafile_name,
                             n_skiprows=n_skiprows)
    current_folder = tbnn_iterate(seed_max=seed_max, x=x, tb=tb, y=y, train_list=train_list, test_list=test_list,
                                  train_valid_random_split=train_valid_random_split, valid_list=valid_list,
                 train_test_random_split=train_test_random_split, train_test_split_fraction=train_test_split_fraction,
                                  n_case_points=n_case_points, num_layers=num_layers, num_nodes=num_nodes, nonlinearity=nonlinearity,
                 nonlinearity_key=nonlinearity_key, nonlinearity_key_value=nonlinearity_key_value,
                 train_valid_split_fraction=train_valid_split_fraction, learning_rate_decay=learning_rate_decay,
                 min_learning_rate=min_learning_rate, weight_initialiser_name=weight_initialiser_name,
                 weight_initialiser_params=weight_initialiser_params, max_epochs=max_epochs, min_epochs=min_epochs,
                 init_learning_rate=init_learning_rate, interval=interval, average_interval=average_interval,
                 loss=loss_function, optimizer=optimizer, batch_size=batch_size,
                 enforce_realizability=enforce_realizability, num_realizability_its=num_realizability_its,
                 folder_path=folder_path)
    stop = timeit.default_timer()
    write_time(start, stop, folder_path, current_folder)
    print("Finish")


if __name__ == "__main__":
    main()



