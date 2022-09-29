"""
In this example, a Tensor Basis Neural Network (TBNN) is trained on the data for F-BFS flows.
The inputs are the mean strain rate tensor Sij and the mean rotation rate tensor Rij (normalized by k and epsilon).
The output is the Reynolds stress anisotropy tensor bij.

Data:
The F-BFS data set is based on 2D data from LES of F-BFS with inlet velocity = 1, 2 and 4 m/s.
The tke, epsilon, and velocity gradients are from k-omega SST RANS simulations for the same flows.

Reference for data driven turbulence modeling TBNN implementation:
Ling, J., Kurzawski, A. and Templeton, J., 2016. Reynolds averaged turbulence modelling using deep neural
networks with embedded invariance. Journal of Fluid Mechanics, 807, pp.155-166.
"""

import os
import timeit
from pred_iterator import preprocessing, trial_iter
from results_writer import write_time


def main():
    # Define parameters
    num_hid_layers = 5  # Number of hidden layers in the TBNN, default = 2
    num_hid_nodes = [5, 5, 5, 5, 5]  # Number of nodes in the hidden layers given as a vector, default = [5, 5, 5, 5, 5]
    num_tensor_basis = 10  # Number of tensor bases to predict, also the num of output nodes, default = 10
    max_epochs = 100000  # Max number of epochs during training, default = 2000
    min_epochs = 60  # Min number of epochs during training, default = 1000
    interval = 10  # Frequency at which convergence is checked, default = 100
    avg_interval = 3  # Number of intervals averaged over for early stopping criteria, default = 4
    enforce_realiz = True  # Boolean for enforcing realizability on Reynolds stresses, default = True
    num_realiz_its = 5  # Number of iterations for realizability enforcing, default = 5

    # Define advanced parameters
    af = "elu"  # Nonlinear activation function, default = "LeakyRectify"
    af_key = ""  # Parameters of the nonlinear activation function, default = "leakiness"
    af_key_value = ""  # Arguments of the nonlinear activation function, default = "0.1"
    weight_init_name = "HeUniform"  # Weight initialiser algorithm, default = "HeUniform"
    weight_init_params = "gain=np.sqrt(2.0)"  # Arguments of the weight initialiser algorithm, default = "gain=np.sqrt(2.0)"
    init_lr = 0.001  # Initial learning rate, default = 0.01
    lr_scheduler = "StepLR"  # Learning rate scheduler
    lr_scheduler_params = "step_size=30, gamma=0.2"  # Parameters of learning rate scheduler
    loss = "MSE"  # Loss function, default = "MSE"
    optimizer = "adam"  # Optimizer algorithm, default = "adam"
    batch_size = 5  # Training batch size, default = 1

    # Define database and data splits for training, validation and testing
    database_name = "FBFS_full_set_no_walls.txt"
    n_skiprows = 1  # Skip this number of rows at the beginning of datafile_name for reading
    train_test_rand_split = False  # Randomly split database for training and testing
    train_test_split_frac = 0  # Fraction of data for training and validation if using random split, default = 0.8
    train_valid_rand_split = False  # Take some % of data in train_list for validation, default = False
    train_valid_split_frac = 0  # Fraction of train data used for training, the other fraction is used for validation, default = 0.9. Train-test split must be run before this.

    # If random splits = False, set case lists
    train_list = [1]
    valid_list = [4]
    test_list = [2]

    # Number of TBNN instances to average predictions over
    n_seeds = 5  # Number of reproducible TBNN predictions to save
    print_freq = 100

    # Write folder
    folder_path = os.path.join(os.getcwd(), "TBNN output data")

    """
    This script runs two main nested functions: preprocessing and trial_iter.
    """
    start = timeit.default_timer()
    x, tb, y = preprocessing(enforce_realiz=enforce_realiz, num_realiz_its=num_realiz_its, database_name=database_name,
                             n_skiprows=n_skiprows, num_tensor_basis=num_tensor_basis)
    user_vars = locals()
    current_folder = trial_iter(n_seeds=n_seeds, x=x, tb=tb, y=y, train_list=train_list, test_list=test_list,
                                  train_valid_rand_split=train_valid_rand_split, valid_list=valid_list,
                 train_test_rand_split=train_test_rand_split, train_test_split_frac=train_test_split_frac,
                                num_hid_layers=num_hid_layers, num_hid_nodes=num_hid_nodes, af=af,
                 af_key=af_key, af_key_value=af_key_value,
                 train_valid_split_frac=train_valid_split_frac, lr_scheduler=lr_scheduler,
                 lr_scheduler_params=lr_scheduler_params, weight_init_name=weight_init_name,
                 weight_init_params=weight_init_params, max_epochs=max_epochs, min_epochs=min_epochs,
                 init_lr=init_lr, interval=interval, avg_interval=avg_interval,
                 loss=loss, optimizer=optimizer, batch_size=batch_size,
                 enforce_realiz=enforce_realiz, num_realiz_its=num_realiz_its,
                 folder_path=folder_path, user_vars=user_vars, print_freq=print_freq, num_tensor_basis=num_tensor_basis,
                                database_name=database_name)
    stop = timeit.default_timer()
    write_time(start, stop, folder_path, current_folder)
    print("Finish")


if __name__ == "__main__":
    main()



