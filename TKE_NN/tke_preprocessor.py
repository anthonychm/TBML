import numpy as np
import sys

sys.path.insert(1, "../TBNN")
from TBNN.preprocessor import DataSplitter


def calc_output(tauij):
    tke = 0.5 * (tauij[:, 0, 0] + tauij[:, 1, 1] + tauij[:, 2, 2])
    tke = np.maximum(tke, 1e-8)
    log_tke = np.log10(tke)

    return log_tke


def split_database(Cx, Cy, x, y, train_list, valid_list, test_list, seed, train_valid_rand_split,
                   train_test_rand_split, train_valid_split_frac, train_test_split_frac, case_dict):
    """
    Split data using the random or specified data-splitting method. The random method randomly allocates (x, tb, y)
    data for TVT. The specified method allocates data from specific cases for TVT according to the TVT lists.
    TVT = training, validation and testing.

    :param x: scalar invariants of Sij and Rij
    :param y: Reynolds stress anisotropy
    :param train_list: list of cases for training (use if train_test_rand_split = False)
    :param valid_list: list of cases for validation (use if train_valid_rand_split = False)
    :param test_list: list of cases for testing (use if train_test_rand_split = False)
    :param seed: current random seed for reproducible database splitting
    :param train_valid_rand_split: boolean for enforcing random train-valid splitting of training data
    :param train_test_rand_split: boolean for enforcing random train-test splitting of all data
    :param train_valid_split_frac: fraction of training data to use for actual training,
    the rest is used for validation (use if train_valid_rand_split = True)
    :param train_test_split_frac: fraction of all data to use for training data (use if train_test_rand_split = True)

    :return: inputs, tensor basis and outputs for the TVT data sets
    """

    def tke_rand_split(Cx, Cy, inputs, outputs, train_idx, test_idx):

        return Cx[train_idx], Cy[train_idx], inputs[train_idx, :], outputs[train_idx, :], \
            Cx[test_idx], Cy[test_idx], inputs[test_idx, :], outputs[test_idx, :]

    def tke_spec_split(Cx, Cy, inputs, outputs, idx):

        return Cx[idx], Cy[idx], inputs[idx, :], outputs[idx, :]

    # Split database randomly if random split = True
    if train_test_rand_split is True:
        train_idx, test_idx = DataSplitter.get_rand_split_idx(x, train_test_split_frac, seed)
        Cx_train, Cy_train, x_train, y_train, Cx_test, Cy_test, x_test, y_test = \
            tke_rand_split(Cx, Cy, x, y, train_idx, test_idx)
        print("Train-test random data split complete")

        if train_valid_rand_split is True:
            train_idx, valid_idx = DataSplitter.get_rand_split_idx(x_train, train_valid_split_frac, seed)
            Cx_train, Cy_train, x_train, y_train, Cx_valid, Cy_valid, x_valid, y_valid = \
                tke_rand_split(Cx, Cy, x_train, y_train, train_idx, valid_idx)
            print("Train-valid random data split complete")
        else:
            Cx_valid, Cy_valid, x_valid, y_valid = [], [], [], []
            print("No data allocated for validation")

    # Else split database according to specified cases if random split = False
    else:
        idx = DataSplitter.get_spec_split_idx(train_list, case_dict, seed, shuffle=True)
        Cx_train, Cy_train, x_train, y_train = tke_spec_split(Cx, Cy, x, y, idx)
        print("Specified data split for training complete")

        idx = DataSplitter.get_spec_split_idx(test_list, case_dict, seed, shuffle=False)
        Cx_test, Cy_test, x_test, y_test = tke_spec_split(Cx, Cy, x, y, idx)
        print("Specified data split for testing complete")

        if train_valid_rand_split is False:
            idx = DataSplitter.get_spec_split_idx(valid_list, case_dict, seed, shuffle=False)
            Cx_valid, Cy_valid, x_valid, y_valid = tke_spec_split(Cx, Cy, x, y, idx)
            print("Specified data split for validation complete")
        else:
            Cx_valid, Cy_valid, x_valid, y_valid = [], [], [], []
            print("No data allocated for validation")

    return Cx_train, Cy_train, x_train, y_train, Cx_test, Cy_test, x_test, y_test, Cx_valid, Cy_valid, x_valid, y_valid
