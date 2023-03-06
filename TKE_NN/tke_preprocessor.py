import numpy as np
import sys

sys.path.insert(1, "../TBNN")
from TBNN.preprocessor import DataSplitter


def calc_output(tauij):  # ✓
    """
    Calculates the logarithm to the base 10 of TKE to ensure that predicted TKE from
    the TKENN is positive.
    """
    tke = 0.5 * (tauij[:, 0, 0] + tauij[:, 1, 1] + tauij[:, 2, 2])
    tke = np.maximum(tke, 1e-8)
    log_tke = np.log10(tke)
    log_tke = np.reshape(log_tke, (log_tke.shape[0], 1))

    return log_tke


def split_database(coords, x, y, train_list, valid_list, test_list, seed,
                   train_valid_rand_split, train_test_rand_split,
                   train_valid_split_frac, train_test_split_frac, case_dict):  # ✓
    """
    Split data using the random or specified data-splitting method. The random method
    randomly allocates (x, y) data for TVT. The specified method allocates data
    from specific cases for TVT according to the TVT lists.
    TVT = training, validation and testing.

    :param coords: coordinates of data rows
    :param x: scalar inputs
    :param y: Reynolds stress anisotropy truth data
    :param train_list: cases for training (use if train_test_rand_split = False)
    :param valid_list: cases for validation (use if train_valid_rand_split = False)
    :param test_list: cases for testing (use if train_test_rand_split = False)
    :param seed: current random seed for reproducible database splitting
    :param train_valid_rand_split: boolean for random train-valid splitting of training
           data
    :param train_test_rand_split: boolean for random train-test splitting of all data
    :param train_valid_split_frac: fraction of training data to use for actual training;
           the rest is used for validation (use if train_valid_rand_split = True)
    :param train_test_split_frac: fraction of all data to use for training and
           validation; the rest is used for testing (use if train_test_rand_split = True)
    :param case_dict: Dictionary specifying number of data points per case

    :return: coordinates, scalar inputs, tensor basis and outputs for the TVT data sets
    """

    def tkenn_rand_split(coords, inputs, outputs, train_idx, test_idx):  # ✓

        return coords[train_idx, :], inputs[train_idx, :], outputs[train_idx], \
            coords[test_idx, :], inputs[test_idx, :], outputs[test_idx]

    def tkenn_spec_split(coords, inputs, outputs, idx):  # ✓

        return coords[idx, :], inputs[idx, :], outputs[idx]

    # Split full database randomly into (train+validation) and test datasets if
    # train_test_rand_split  = True ✓
    if train_test_rand_split is True:
        train_idx, test_idx = \
            DataSplitter.get_rand_split_idx(x, train_test_split_frac, seed)  # ✓
        coords_train, x_train, y_train, coords_test, x_test, y_test = \
            tkenn_rand_split(coords, x, y, train_idx, test_idx)  # ✓
        print("Train-test random data split complete")

        # Split (train+validation) dataset randomly into training and validation
        # datasets if train_valid_rand_split = True ✓
        if train_valid_rand_split is True:
            train_idx, valid_idx = \
                DataSplitter.get_rand_split_idx(x_train, train_valid_split_frac, seed)  # ✓
            coords_train, x_train, y_train, coords_valid, x_valid, y_valid = \
                tkenn_rand_split(coords_train, x_train, y_train, train_idx, valid_idx)  # ✓
            print("Train-valid random data split complete")
        else:
            coords_valid, x_valid, y_valid = [], [], []
            print("No data allocated for validation")

    # Else split full database according to specified cases if train_test_rand_split =
    # False ✓
    else:
        # Specified training database ✓
        idx = DataSplitter.get_spec_split_idx(train_list, case_dict, seed,
                                              shuffle=True)  # ✓
        coords_train, x_train, y_train = tkenn_spec_split(coords, x, y, idx)  # ✓
        print("Specified data split for training complete")

        # Specified testing database ✓
        idx = DataSplitter.get_spec_split_idx(test_list, case_dict, seed,
                                              shuffle=False)  # ✓
        coords_test, x_test, y_test = tkenn_spec_split(coords, x, y, idx)  # ✓
        print("Specified data split for testing complete")

        # Specified validation database ✓
        if train_valid_rand_split is False:
            idx = DataSplitter.get_spec_split_idx(valid_list, case_dict, seed,
                                                  shuffle=False)  # ✓
            coords_valid, x_valid, y_valid = tkenn_spec_split(coords, x, y, idx)  # ✓
            print("Specified data split for validation complete")
        else:
            coords_valid, x_valid, y_valid = [], [], []
            print("No data allocated for validation")

    return coords_train, x_train, y_train, coords_test, x_test, y_test, coords_valid, \
        x_valid, y_valid
