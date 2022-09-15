import _pickle as pickle
import random
import numpy as np

"""
Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000,
there is a non-exclusive license for use of this work by or on behalf of the U.S. Government.
This software is distributed under the BSD-3-Clause license.
"""


class DataProcessor:
    """
    Parent class for data processing
    """
    def __init__(self):
        self.mu = None
        self.std = None

    def calc_scalar_basis(self, input_tensors, is_train=False, *args, **kwargs):
        if is_train is True or self.mu is None or self.std is None:
            print("Re-setting normalization constants")

    def calc_tensor_basis(self, input_tensors, *args, **kwargs):
        pass

    def calc_output(self, outputs, *args, **kwargs):
        return outputs

    @staticmethod
    def train_test_split(inputs, tb, outputs, fraction=0.8, randomize=True, seed=None):
        """
        Randomly splits CFD data into training and test set
        :param inputs: scalar invariants
        :param tb: tensor basis
        :param outputs: Reynolds stress anisotropy
        :param fraction: fraction of all CFD data to use for training data
        :param randomize: if True, randomly shuffles data along first axis before splitting it
        :param seed: Random seed for reproducible train-test splitting
        :return: inputs, tb and outputs for the training and test sets
        """
        num_points = inputs.shape[0]
        assert 0 <= fraction <= 1, "fraction must be a real number between 0 and 1"
        num_train = int(fraction*num_points)
        idx = list(range(num_points))
        if randomize:
            if seed:
                random.seed(seed)
            random.shuffle(idx)
        train_idx = idx[:num_train]
        test_idx = idx[num_train:]
        print('random train_test_split complete')
        return inputs[train_idx, :], tb[train_idx, :, :], outputs[train_idx, :], \
               inputs[test_idx, :], tb[test_idx, :, :], outputs[test_idx, :]

    @staticmethod
    def train_test_specified(inputs, tb, outputs, train_list, test_list, train_valid_random_split, valid_list, n_case_points):
        """
        Splits the x, tb and y arrays into train and test sets using the specified train and test cases
        :param inputs: scalar invariants
        :param tb: tensor basis
        :param outputs: Reynolds stress anisotropy
        :param train_list: list of cases for training
        :param test_list: list of cases for testing
        :param n_case_points: Number of data points per CFD case
        :return: inputs, tb and outputs for the training and test sets
        """

        case_dict = {1: 1, 2: 2, 4: 3}
        x_train = np.empty((0, 5))
        tb_train = np.empty((0, 10, 9))
        y_train = np.empty((0, 9))
        x_test = np.empty((0, 5))
        tb_test = np.empty((0, 10, 9))
        y_test = np.empty((0, 9))

        for case in train_list:
            extract_rows = list(range(0, case_dict[case]*n_case_points))
            x_train = np.concatenate((x_train, inputs[extract_rows[-n_case_points:], :]), axis = 0)
            tb_train = np.concatenate((tb_train, tb[extract_rows[-n_case_points:], :, :]), axis = 0)
            y_train = np.concatenate((y_train, outputs[extract_rows[-n_case_points:], :]), axis = 0)

        for case in test_list:
            extract_rows = list(range(0, case_dict[case]*n_case_points))
            x_test = np.concatenate((x_test, inputs[extract_rows[-n_case_points:], :]), axis = 0)
            tb_test = np.concatenate((tb_test, tb[extract_rows[-n_case_points:], :, :]), axis = 0)
            y_test = np.concatenate((y_test, outputs[extract_rows[-n_case_points:], :]), axis = 0)

        x_valid, tb_valid, y_valid = [], [], []

        if train_valid_random_split is False:
            x_valid = np.empty((0, 5))
            tb_valid = np.empty((0, 10, 9))
            y_valid = np.empty((0, 9))

            for case in valid_list:
                extract_rows = list(range(0, case_dict[case] * n_case_points))
                x_valid = np.concatenate((x_valid, inputs[extract_rows[-n_case_points:], :]), axis=0)
                tb_valid = np.concatenate((tb_valid, tb[extract_rows[-n_case_points:], :, :]), axis=0)
                y_valid = np.concatenate((y_valid, outputs[extract_rows[-n_case_points:], :]), axis=0)

        print('specified train_test_split complete')
        return x_train, tb_train, y_train, x_test, tb_test, y_test, x_valid, tb_valid, y_valid

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))
