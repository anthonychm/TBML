import random
import numpy as np


def load_data(database, num_dims, num_input_markers, num_zonal_markers, pressure_tf,
              tke_tf, input_markers_tf, zonal_markers_tf):  # ✓
    """
    Load CFD data for TBNN calculations.
    :param database: Numpy array containing the full database. The columns must be in the
    following order: coordinates, tke, epsilon, velocity gradients, pressure gradients*,
    velocity*, tke gradients*, input markers*, zonal markers* and Reynolds stresses†.

    :param num_dims: Number of dimensions in the data
    :param num_input_markers: Number of scalar input markers
    :param num_zonal_markers: Number of scalar zonal markers
    :param pressure_tf: Boolean for including pressure gradient invariants in TBNN input
    :param tke_tf: Boolean for including tke gradient invariants in TBNN input
    :param input_markers_tf: Boolean for including scalar input markers in TBNN input
    :param zonal_markers_tf: Boolean for including scalar zonal markers for zonal approach

    :return: From RANS: Coordinates, tke, epsilon, velocity gradients, pressure
    gradients*, velocity*, tke gradients*, scalar input markers*.
    From LES/DNS/experiment: Flattened Reynolds stress tensor.

    * optional; pressure gradients and velocity are together.
    † from LES/DNS/experiment; all other columns are from RANS.
    """

    # For 2D:
    # Var  | Cx | Cy | k | eps | gradU | gradp |  U  | gradk |   marker inputs   |
    # Col     0    1   2    3     4-12   13-15  16-18  19-21   -(nm+nz+9):-(nz+9)

    # Var |  zonal marker  |tauij|
    # Col     -(nz+9):-9     -9:

    # Define compulsory data ✓
    coords = database[:, :num_dims]
    k = database[:, num_dims]
    eps = database[:, num_dims + 1]
    grad_u_flat = database[:, num_dims + 2:num_dims + 11]
    tauij_flat = database[:, -9:]

    # Define grad_p data ✓
    if pressure_tf is True:
        grad_p = database[:, num_dims + 11:num_dims + 14]
        u = database[:, num_dims + 14:num_dims + 17]
    else:
        grad_p = float("nan")
        u = float("nan")

    # Define grad_k data ✓
    if pressure_tf is False and tke_tf is True:
        grad_k = database[:, num_dims + 11:num_dims + 14]
    elif pressure_tf is True and tke_tf is True:
        grad_k = database[:, num_dims + 17:num_dims + 20]
    else:
        grad_k = float("nan")

    # Define input markers data [UNCHECKED]
    if input_markers_tf is True and zonal_markers_tf is False:
        input_markers = database[:, -(num_input_markers+9):-9]
    elif input_markers_tf is True and zonal_markers_tf is True:
        input_markers = \
            database[:, -(num_input_markers+num_zonal_markers+9):-(num_zonal_markers+9)]
    else:
        input_markers = float("nan")

    # Reshape grad_u and stresses to num_points X 3 X 3 arrays ✓
    num_points = database.shape[0]
    grad_u = np.zeros((num_points, 3, 3))
    for i in range(3):
        for j in range(3):
            grad_u[:, i, j] = grad_u_flat[:, (3*i)+j]

    return coords, k, eps, grad_u, grad_p, u, grad_k, input_markers, tauij_flat


def scalar_basis_manager(data_processor, k, eps, grad_u, rho, u, grad_p, grad_k,
                         two_invars, incl_p_invars, incl_tke_invars):  # ✓

    Sij, Rij = data_processor.calc_Sij_Rij(grad_u, k, eps)  # ✓
    # Mean strain rate tensor Sij = (k/eps)*sij, mean rotation rate tensor (k/eps)*rij

    if incl_p_invars is True:
        Ap = data_processor.calc_Ap(grad_p, rho, u, grad_u)  # ✓
    if incl_tke_invars is True:
        Ak = data_processor.calc_Ak(grad_k, eps, k)  # ✓

    if two_invars is True:
        x = data_processor.calc_scalar_basis(Sij, Rij, pressure=False, tke=False,
                                             two_invars=True)
        assert x.shape[1] == 2, "Incorrect number of columns"
    elif incl_p_invars is False and incl_tke_invars is False:
        x = data_processor.calc_scalar_basis(Sij, Rij, pressure=False, tke=False)  # ✓
        assert x.shape[1] == 5, "Incorrect number of columns"
    elif incl_p_invars is True and incl_tke_invars is False:
        x = data_processor.calc_scalar_basis(Sij, Rij, Ap=Ap, pressure=True, tke=False)
        # ✓
        assert x.shape[1] == 19, "Incorrect number of columns"
    elif incl_p_invars is False and incl_tke_invars is True:
        x = data_processor.calc_scalar_basis(Sij, Rij, Ak=Ak, pressure=False, tke=True)
        # ✓
        assert x.shape[1] == 19, "Incorrect number of columns"
    elif incl_p_invars is True and incl_tke_invars is True:
        x = data_processor.calc_scalar_basis(Sij, Rij, Ap=Ap, Ak=Ak, pressure=True,
                                             tke=True)  # ✓
        assert x.shape[1] == 47, "Incorrect number of columns"

    return Sij, Rij, x


class DataSplitter:
    """
    Class for splitting the entire database into TVT databases
    """

    @staticmethod
    def get_rand_split_idx(inputs, fraction, seed):  # ✓
        """
        Randomly splits CFD data into training and test dataset
        :param inputs: scalar inputs
        :param fraction: fraction of all CFD data to use for training data
        :param seed: random seed for reproducible train-test splitting
        :return: row indexes corresponding to the training and test datasets
        """
        num_points = inputs.shape[0]
        assert 0 <= fraction <= 1, "Fraction must be a real number between 0 and 1"
        num_train = int(fraction * num_points)
        idx = list(range(num_points))
        idx = DataSplitter.shuffle_idx(seed, idx)  # ✓
        train_idx = idx[:num_train]
        test_idx = idx[num_train:]

        return train_idx, test_idx

    @staticmethod
    def tbnn_rand_split(coords, inputs, tensor_basis, outputs, train_idx, test_idx):  # ✓

        return coords[train_idx, :], inputs[train_idx, :], \
            tensor_basis[train_idx, :, :], outputs[train_idx, :], coords[test_idx, :], \
            inputs[test_idx, :], tensor_basis[test_idx, :, :], outputs[test_idx, :]

    @staticmethod
    def get_spec_split_idx(case_list, case_dict, seed, shuffle=False):  # ✓

        # Loop through cases in list to extract idx ✓
        idx = []
        for case in case_list:
            keys = [key for key in case_dict.keys() if key <= case]
            end_row = 0
            for key in keys:
                end_row += case_dict[key]

            extract_rows = list(range(0, end_row))
            idx.extend(extract_rows[-case_dict[case]:])

        idx = DataSplitter.shuffle_idx(seed, idx, shuffle)  # ✓

        return idx

    @staticmethod
    def tbnn_spec_split(coords, inputs, tensor_basis, outputs, idx):  # ✓

        return coords[idx, :], inputs[idx, :], tensor_basis[idx, :, :], outputs[idx, :]

    @staticmethod
    def shuffle_idx(seed, idx, shuffle=True):  # ✓
        if shuffle:
            if seed:
                random.seed(seed)
            random.shuffle(idx)

        return idx
