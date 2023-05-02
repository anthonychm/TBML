import numpy as np
import random
import zonal_marker_calculator as zmc


class ZonalSplitter:
    def __init__(self, case_list, coords_list, marker_list, zones):
        # Repeat for train, valid and test
        self.case_list = case_list
        self.num_rows_list = []
        self.marker_dict = {k: [] for k in coords_list + marker_list}
        self.row_idx_dict = {k: [] for k in zones}

    def assemble_marker_dict(self, coords_list, marker_list):
        for case in self.case_list:
            parent_path = zmc.get_parent_path(case)
            coords_dict = zmc.load_marker_data_apr2023(case, parent_path, coords_list)
            self.num_rows_list.append(len(coords_dict["Cx"]))
            for coord in coords_list:
                self.marker_dict[coord].extend(coords_dict[coord])

            marker_dict = zmc.load_zonal_markers_apr2023(case, parent_path, marker_list)
            for marker in marker_list:
                self.marker_dict[marker].extend(marker_dict[marker])

    def apply_nd_Q_marker(self, i, allocate_zone):
        # Apply non-dimensional Q-criterion zonal marker
        if self.marker_dict["nd_Q"][i] > 0:
            self.row_idx_dict["nd_Q"].append(i)
            allocate_zone = False
        return allocate_zone

    def apply_nd_TI_marker(self, i, allocate_zone):
        # Apply non-dimensional turbulence intensity zonal marker
        if self.marker_dict["nd_TI"][i] > 0.5:
            self.row_idx_dict["nd_TI"].append(i)
            allocate_zone = False
        return allocate_zone

    def apply_nd_Ux_marker(self, i, allocate_zone):
        # Apply non-dimensional streamwise velocity zonal marker
        if self.marker_dict["nd_Ux"][i] < 0:
            self.row_idx_dict["nd_Ux"].append(i)
            allocate_zone = False
        return allocate_zone

    def apply_lam_marker(self, i, allocate_zone):
        # Apply laminar marker (only to FBFS cases in Apr 2023)
        if self.marker_dict["Cx"][i] < -0.003:
            self.row_idx_dict["lam"].append(i)
            allocate_zone = False
        return allocate_zone

    def apply_Re_y_marker(self, i, allocate_zone):
        # Apply turbulence Reynolds number zonal marker
        if self.marker_dict["Re_y"][i] < 1:
            self.row_idx_dict["Re_y"].append(i)
            allocate_zone = False
        return allocate_zone

    def allocate_zone(self):
        # Apply the zonal markers to fill the row idx dictionary (i.e. allocate a zone)
        for i in range(sum(self.num_rows_list)):
            allocate_zone = True
            while allocate_zone:
                allocate_zone = self.apply_nd_Q_marker(i, allocate_zone)
                if allocate_zone is False:
                    break
                allocate_zone = self.apply_nd_TI_marker(i, allocate_zone)
                if allocate_zone is False:
                    break
                allocate_zone = self.apply_nd_Ux_marker(i, allocate_zone)
                if allocate_zone is False:
                    break
                # allocate_zone = self.apply_lam_marker(i, allocate_zone)
                # if allocate_zone is False:
                    # break
                allocate_zone = self.apply_Re_y_marker(i, allocate_zone)
                if allocate_zone is False:
                    break
                else:
                    self.row_idx_dict["fstrm"].append(i)
                    break


class ZonalDataConcatenator:
    def __init__(self, case_list, row_idx_dict, num_rows_list):
        self.case_list = case_list
        self.row_idx_dict = row_idx_dict
        self.num_rows_list = num_rows_list

    def init_zonal_db_dict(self, zones, num_cols):
        # Initialise zonal database dictionary
        zonal_db_dict = {zone: np.full((len(self.row_idx_dict[zone]), num_cols), np.nan)
                         for zone in zones}
        return zonal_db_dict

    def concat_zonal_data(self, zones, num_cols):
        # Initialise zonal database dictionary, then loop for each zone and each row index
        zonal_db_dict = self.init_zonal_db_dict(zones, num_cols)
        for zone in self.row_idx_dict:
            case_idx, next_case_row_idx = -1, 0
            for count, row_idx in enumerate(self.row_idx_dict[zone]):

                if row_idx >= next_case_row_idx:
                    # Update case data
                    case_idx += 1
                    datum_row_idx = next_case_row_idx
                    next_case_row_idx += self.num_rows_list[case_idx]
                    data_dict, true_tauij = load_data(self.case_list[case_idx])

                    # Flatten gradU and tauij arrays
                    if len(data_dict["gradU"].shape) == 3:
                        gradU = flatten(data_dict["gradU"])
                        data_dict["gradU"] = gradU
                    if len(true_tauij.shape) == 3:
                        true_tauij = flatten(true_tauij)

                # Fill zonal database dictionary row
                i = row_idx - datum_row_idx
                row_data = np.hstack((data_dict["Cx"][i], data_dict["Cy"][i],
                                      data_dict["k"][i], data_dict["epsilon"][i],
                                      data_dict["gradU"][i, :], true_tauij[i, :]))
                zonal_db_dict[zone][count, :] = row_data
        return zonal_db_dict


def load_true_tauij(parent_path, case):
    if "FBFS" in case:
        true_tauij = zmc.create_var_dict(["LES_tau"], parent_path, "/" + case + "/", case)
        true_tauij = true_tauij["LES_tau"]
    elif any(name in case for name in ["BUMP", "CBFS", "CNDV", "PHLL", "DUCT"]):
        true_tauij = zmc.create_var_dict(["tau"], parent_path, "/labels/", case)
        true_tauij = true_tauij["tau"]
    else:
        raise Exception("No matching true tauij for this case")
    return true_tauij


def load_data(case):
    parent_path = zmc.get_parent_path(case)
    data_dict = zmc.load_marker_data_apr2023(case, parent_path,
                                             ["Cx", "Cy", "k", "epsilon", "gradU"])
    true_tauij = load_true_tauij(parent_path, case)
    return data_dict, true_tauij


def flatten(array):
    flat_array = np.full((array.shape[0], 9), np.nan)
    for row in range(array.shape[0]):
        flat_array[row, :] = np.hstack((array[row, 0, :], array[row, 1, :],
                                        array[row, 2, :]))
    # assert(np.isnan(flat_array).any() is False)
    return flat_array


def shuffle_rows(zonal_db_dict, seed):
    # Shuffle concatenated zonal data rows [ONLY APPLY THIS TO TRAINING DATASET]
    for zone in zonal_db_dict:
        train_data = zonal_db_dict[zone]
        idx = list(range(train_data.shape[0]))
        random.seed(seed)
        random.shuffle(idx)
        zonal_db_dict[zone] = train_data[idx, :]
    return zonal_db_dict


def create_zonal_db(case_list, coords_list, marker_list, zones, num_cols):
    # Workflow for creating a zonal database (repeat for training, validation and testing)
    zsplitter = ZonalSplitter(case_list, coords_list, marker_list, zones)
    zsplitter.assemble_marker_dict(coords_list, marker_list)
    zsplitter.allocate_zone()
    zcatr = ZonalDataConcatenator(case_list, zsplitter.row_idx_dict,
                                  zsplitter.num_rows_list)
    zonal_db_dict = zcatr.concat_zonal_data(zones, num_cols)
    return zonal_db_dict


def create_non_zonal_db(case_list):
    # Workflow for creating a non-zonal database (repeat for training, validation and
    # testing
    for count, case in enumerate(case_list):
        data_dict, true_tauij = load_data(case)

        # Flatten gradU and tauij arrays
        if len(data_dict["gradU"].shape) == 3:
            gradU = flatten(data_dict["gradU"])
            data_dict["gradU"] = gradU
        if len(true_tauij.shape) == 3:
            true_tauij = flatten(true_tauij)

        # Concatenate case data and combine with previous case data
        case_data = np.hstack((data_dict["Cx"], data_dict["Cy"], data_dict["k"],
                               data_dict["epsilon"], data_dict["gradU"], true_tauij))
        if count == 0:
            non_zonal_db = case_data
        else:
            non_zonal_db = np.vstack((non_zonal_db, case_data))

    # Convert database array into dictionary
    non_zonal_db_dict = {["non_zonal"]: non_zonal_db}
    return non_zonal_db_dict
