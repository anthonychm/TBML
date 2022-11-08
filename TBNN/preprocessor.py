import random
import numpy as np


def load_data(database, num_input_markers, num_zonal_markers=float("nan"), pressure_tf=False, tke_tf=False,
              input_markers_tf=False, zonal_markers_tf=False):
    """
    Load CFD data for TBNN calculations.
    :param database_name: Name of the data file containing the CFD data. The columns must be in the following order: TKE
    from RANS, epsilon from RANS, velocity gradients from RANS and Reynolds stresses from high-fidelity simulation.
    :param n_skiprows: Number of rows at top of data file which this code should skip for reading
    :return: Variables k and eps from RANS, flattened grad_u tensor from RANS and flattened Reynolds stress tensor from
    high-fidelity simulation.
    """

    # Var  Cx | Cy | k | eps | gradU | gradp |  U  | gradk |   marker inputs   |  zonal marker  |tauij|
    # Col   0    1   2    3     4-12   13-15  16-18  19-21   -(nm+nz+9):-(nz+9)    -(nz+9):-9     -9:

    # Define compulsory data
    Cx = database[:, 0]
    Cy = database[:, 1]
    k = database[:, 2]
    eps = database[:, 3]
    grad_u_flat = database[:, 4:13]
    tauij_flat = database[:, -9:]

    # Define grad_p data
    if pressure_tf is True:
        grad_p = database[:, 13:16]
        u = database[:, 16:19]
    else:
        grad_p = float("nan")
        u = float("nan")

    # Define grad_k data
    if pressure_tf is False and tke_tf is True:
        grad_k = database[:, 13:16]
    elif pressure_tf is True and tke_tf is True:
        grad_k = database[:, 19:22]
    else:
        grad_k = float("nan")

    # Define input markers data
    if input_markers_tf is True and zonal_markers_tf is False:
        input_markers = database[:, -(num_input_markers+9):-9]
    elif input_markers_tf is True and zonal_markers_tf is True:
        input_markers = database[:, -(num_input_markers+num_zonal_markers+9):-(num_zonal_markers+9)]
    else:
        input_markers = float("nan")

    # Reshape grad_u and stresses to num_points X 3 X 3 arrays
    num_points = database.shape[0]
    grad_u = np.zeros((num_points, 3, 3))
    tauij = np.zeros((num_points, 3, 3))
    for i in range(3):
        for j in range(3):
            grad_u[:, i, j] = grad_u_flat[:, i*3+j] #TO DO: Check
            tauij[:, i, j] = tauij_flat[:, i*3+j] #TO DO: Check

    return Cx, Cy, k, eps, grad_u, grad_p, u, grad_k, input_markers, tauij


def scalar_basis_manager(data_processor, k, eps, grad_u, rho, u, grad_p, grad_k, incl_p_invars=False,
                         incl_tke_invars=False):

    Sij, Rij = data_processor.calc_Sij_Rij(grad_u, k, eps)  # Mean strain rate tensor Sij = k/eps*Sij,
    # mean rotation rate tensor k/eps*Rij

    if incl_p_invars is True:
        Ap = data_processor.calc_Ap(grad_p, rho, u, grad_u)
    elif incl_tke_invars is True:
        Ak = data_processor.calc_Ak(grad_k, eps, k)

    if incl_p_invars is False and incl_tke_invars is False:
        x = data_processor.calc_scalar_basis(Sij, Rij, pressure=False, tke=False)
    elif incl_p_invars is True and incl_tke_invars is False:
        x = data_processor.calc_scalar_basis(Sij, Rij, Ap=Ap, pressure=True, tke=False)
    elif incl_p_invars is False and incl_tke_invars is True:
        x = data_processor.calc_scalar_basis(Sij, Rij, Ak=Ak, pressure=False, tke=True)
    elif incl_p_invars is True and incl_tke_invars is True:
        x = data_processor.calc_scalar_basis(Sij, Rij, Ap=Ap, Ak=Ak, pressure=True, tke=True)

    return Sij, Rij, x


class DataSplitter:
    """
    Class for splitting the entire database into training, validation and testing databases
    """

    @staticmethod
    def random_split(Cx, Cy, inputs, tb, outputs, fraction, seed, shuffle=True):
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
        DataSplitter.shuffle_idx(seed, idx, shuffle)
        train_idx = idx[:num_train]
        test_idx = idx[num_train:]

        return Cx[train_idx], Cy[train_idx], inputs[train_idx, :], tb[train_idx, :, :], outputs[train_idx, :], \
            Cx[test_idx], Cy[test_idx], inputs[test_idx, :], tb[test_idx, :, :], outputs[test_idx, :]

    @staticmethod
    def specified_split(Cx, Cy, inputs, tensor_basis, outputs, case_list, case_dict, num_tensor_basis, seed,
                        shuffle=True):
        """
        Splits the x, tb and y arrays into train and test sets using the specified train and test cases
        :param inputs: scalar invariants
        :param tensor_basis: tensor basis
        :param outputs: Reynolds stress anisotropy
        :param case_list: list of cases for extraction
        :param case_dict: dictionary for extracting case index
        :return: inputs, tb and outputs for the training and test sets
        """

        # Initialise x, tb and y arrays
        x = np.empty((0, 5))
        tb = np.empty((0, num_tensor_basis, 9))
        y = np.empty((0, 9))

        # Loop through cases in list to extract x, tb and y
        for case in case_list:
            keys = [key for key in case_dict.keys() if key <= case]
            end_row = 0
            for key in keys:
                end_row += case_dict[key]

            extract_rows = list(range(0, end_row))
            x = np.concatenate((x, inputs[extract_rows[-case_dict[case]:], :]), axis=0)
            tb = np.concatenate((tb, tensor_basis[extract_rows[-case_dict[case]:], :, :]), axis=0)
            y = np.concatenate((y, outputs[extract_rows[-case_dict[case]:], :]), axis=0)

        num_points = x.shape[0]
        assert x.shape[0] == tb.shape[0] == y.shape[0], "Mismatch in number of data points"
        idx = list(range(num_points))
        DataSplitter.shuffle_idx(seed, idx, shuffle)

        return Cx[idx], Cy[idx], x[idx, :], tb[idx, :, :], y[idx, :]

    @staticmethod
    def shuffle_idx(seed, idx, shuffle=True):
        if shuffle:
            if seed:
                random.seed(seed)
            random.shuffle(idx)

        return idx
