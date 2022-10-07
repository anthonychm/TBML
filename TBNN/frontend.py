"""
===============================================
= This code has been written by Anthony Man   =
= PhD student at The University of Manchester =
===============================================

This program performs all the steps for training and testing a Tensor Basis Neural Network (TBNN) for any kind of
turbulent flow problem.

Input 1: Five invariants of Sij and Rij
Input 2: Tensor basis of Sij and Rij
Sij = Mean strain rate tensor, Rij = Mean rotation rate tensor
Both Sij and Rij are from RANS simulations and normalized by k and epsilon
Output: Reynolds stress anisotropy tensor bij
True bij is from highly-resolved LES or DNS. The TBNN aims to produce this true output.

Reference for data driven turbulence modelling TBNN implementation:
Ling, J., Kurzawski, A. and Templeton, J., 2016. Reynolds averaged turbulence modelling using deep neural
networks with embedded invariance. Journal of Fluid Mechanics, 807, pp.155-166.
"""

import timeit
from pred_iterator import preprocessing, trial_iter
from results_writer import write_time, create_parent_folders


def main():
    # Define parameters
    num_hid_layers = 3  # Number of hidden layers in the TBNN, default = 2
    num_hid_nodes = [7, 8, 2]  # Number of nodes in the hidden layers given as a vector, default = [5, 5, 5, 5, 5]
    num_tensor_basis = 3  # Number of tensor bases to predict, also the num of output nodes, default = 10
    max_epochs = 50  # Max number of training epochs, default = 2000
    min_epochs = 10  # Min number of training epochs, default = 1000
    interval = 2  # Frequency at which convergence is checked, default = 100
    avg_interval = 3  # Number of intervals averaged over for early stopping criteria, default = 4
    enforce_realiz = True  # Boolean for enforcing realizability on Reynolds stresses, default = True ##
    num_realiz_its = 5  # Number of iterations for realizability enforcing, default = 5

    # Define advanced parameters
    af = ["LeakyReLU", "ELU", "LeakyReLU"]
    # af = ["ELU"]*(num_hid_layers-1)  # Nonlinear activation function, default = ["ELU"]*(num_hid_layers-1)
    af_params = ["negative_slope=0.012", "alpha=0.8", "negative_slope=0.009"]
    #af_params = ["alpha=1.0, inplace=False"]*(num_hid_layers-1)  # Parameters of the nonlinear activation function,
                                                             # default = ["alpha=1.0, inplace=False"]*(num_hid_layers-1)
    weight_init = "xavier_uniform_"  # Weight initialiser algorithm, default = "xavier_uniform"
    weight_init_params = "gain=1"  # Arguments of the weight initialiser algorithm, default = "gain=1.0"
    init_lr = 0.001  # Initial learning rate, default = 0.01
    lr_scheduler = "ExponentialLR"  # Learning rate scheduler, default = ExponentialLR
    lr_scheduler_params = "gamma=0.1"  # Parameters of learning rate scheduler, default = "gamma=0.1"
    loss = "MSELoss"  # Loss function, default = "MSELoss"
    optimizer = "Adam"  # Optimizer algorithm, default = "Adam"
    batch_size = 5  # Training batch size, default = 1

    # Define database and data splits for training, validation and testing
    database_name = "full_set_no_walls.txt"  # Data source
    n_skiprows = 1  # Number of rows to skip at beginning of data source when reading
    train_test_rand_split = False  # Boolean for randomly splitting entire database for training and testing,
                                   # default = False
    train_test_split_frac = 0  # Fraction of entire database used for training and validation if random split = True,
                               # default = 0
    train_valid_rand_split = False  # Boolean for randomly splitting training database for training and validation,
                                    # default = False
    train_valid_split_frac = 0  # Fraction of training database used for actual training, the other fraction is used
                                # for validation, default = 0. Train-test split must be run before this.

    # Set case lists if train_test_rand_split and train_valid_rand_split = False
    train_list = [180, 290, 490, 760, 945]
    valid_list = [395]
    test_list = [590]

    # Other
    n_seeds = 5  # Number of reproducible TBNN predictions to save
    print_freq = 2  # Console print frequency

    folder_path = create_parent_folders()
    start = timeit.default_timer()
    x, tb, y = preprocessing(enforce_realiz=enforce_realiz, num_realiz_its=num_realiz_its, database_name=database_name,
                             n_skiprows=n_skiprows, num_tensor_basis=num_tensor_basis)
    user_vars = locals()
    current_folder = \
        trial_iter(n_seeds=n_seeds, x=x, tb=tb, y=y, train_list=train_list, valid_list=valid_list, test_list=test_list,
                   train_valid_rand_split=train_valid_rand_split, train_valid_split_frac=train_valid_split_frac,
                   train_test_rand_split=train_test_rand_split, train_test_split_frac=train_test_split_frac,
                   num_tensor_basis=num_tensor_basis, num_hid_layers=num_hid_layers, num_hid_nodes=num_hid_nodes, af=af,
                   af_params=af_params, init_lr=init_lr, lr_scheduler=lr_scheduler,
                   lr_scheduler_params=lr_scheduler_params, weight_init=weight_init,
                   weight_init_params=weight_init_params, max_epochs=max_epochs, min_epochs=min_epochs,
                   interval=interval, avg_interval=avg_interval, loss=loss, optimizer=optimizer, batch_size=batch_size,
                   enforce_realiz=enforce_realiz, num_realiz_its=num_realiz_its, folder_path=folder_path,
                   user_vars=user_vars, print_freq=print_freq, database_name=database_name)
    stop = timeit.default_timer()
    write_time(start, stop, folder_path, current_folder)
    print("Finish")


if __name__ == "__main__":
    main()



