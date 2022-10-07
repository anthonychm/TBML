import random
import numpy as np


def load_data(database_name, n_skiprows):
    """
    Load CFD data for TBNN calculations.
    :param database_name: Name of the data file containing the CFD data. The columns must be in the following order: TKE
    from RANS, epsilon from RANS, velocity gradients from RANS and Reynolds stresses from high-fidelity simulation.
    :param n_skiprows: Number of rows at top of data file which this code should skip for reading
    :return: Variables k and eps from RANS, flattened grad_u tensor from RANS and flattened Reynolds stress tensor from
    high-fidelity simulation.
    """

    # Load in database and separate data
    data = np.loadtxt(database_name, skiprows=n_skiprows)
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


class DataProcessor:
    """
    Parent class for data processing
    """
    def __init__(self):
        self.mu = None
        self.std = None

    def calc_scalar_basis(self, Sij, Rij, is_train=False, *args, **kwargs):
        if is_train is True or self.mu is None or self.std is None:
            print("Re-setting normalization constants")

    @staticmethod
    def calc_tensor_basis(Sij, Rij, *args, **kwargs):
        pass

    @staticmethod
    def calc_output(tauij):
        return tauij

    @staticmethod
    def random_split(inputs, tb, outputs, fraction=0.8, shuffle=True, seed=None):
        """
        Randomly splits CFD data into training and test set
        :param inputs: scalar invariants
        :param tb: tensor basis
        :param outputs: Reynolds stress anisotropy
        :param fraction: fraction of all CFD data to use for training data
        :param shuffle: if True, randomly shuffles data along first axis before splitting it
        :param seed: Random seed for reproducible train-test splitting
        :return: inputs, tb and outputs for the training and test sets
        """
        num_points = inputs.shape[0]
        assert 0 <= fraction <= 1, "Fraction must be a real number between 0 and 1"
        num_train = int(fraction*num_points)
        idx = list(range(num_points))
        DataProcessor.shuffle_idx(seed, idx, shuffle)
        train_idx = idx[:num_train]
        test_idx = idx[num_train:]

        return inputs[train_idx, :], tb[train_idx, :, :], outputs[train_idx, :], \
               inputs[test_idx, :], tb[test_idx, :, :], outputs[test_idx, :]

    @staticmethod
    def specified_split(inputs, tensor_basis, outputs, case_list, n_case_points, case_dict, num_tensor_basis, seed,
                        shuffle=True):
        """
        Splits the x, tb and y arrays into train and test sets using the specified train and test cases
        :param inputs: scalar invariants
        :param tensor_basis: tensor basis
        :param outputs: Reynolds stress anisotropy
        :param case_list: list of cases for extraction
        :param n_case_points: Number of data points per CFD case
        :param case_dict: dictionary for extracting case index
        :return: inputs, tb and outputs for the training and test sets
        """

        # Initialise x, tb and y arrays
        x = np.empty((0, 5))
        tb = np.empty((0, num_tensor_basis, 9))
        y = np.empty((0, 9))

        # Loop through cases in list to extract x, tb and y
        for case in case_list:
            extract_rows = list(range(0, case_dict[case]*n_case_points))
            x = np.concatenate((x, inputs[extract_rows[-n_case_points:], :]), axis=0)
            tb = np.concatenate((tb, tensor_basis[extract_rows[-n_case_points:], :, :]), axis=0)
            y = np.concatenate((y, outputs[extract_rows[-n_case_points:], :]), axis=0)

        num_points = x.shape[0]
        assert x.shape[0] == tb.shape[0] == y.shape[0], "Mismatch in number of data points"
        idx = list(range(num_points))
        DataProcessor.shuffle_idx(seed, idx, shuffle)

        return x[idx, :], tb[idx, :, :], y[idx, :]

    @staticmethod
    def shuffle_idx(seed, idx, shuffle=True):
        if shuffle:
            if seed:
                random.seed(seed)
            random.shuffle(idx)

        return idx

