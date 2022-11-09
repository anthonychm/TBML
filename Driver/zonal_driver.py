import numpy as np
import sys
import timeit

from TBNN import frontend, results_writer, case_dicts
from TKE_NN import tke_frontend


def zonal_driver():
    sys.path.insert(1, "../TBNN")
    sys.path.insert(2, "../TKE_NN")

    # Load database and associated dictionary
    database_name = "full_set_no_walls.txt"
    db2case = case_dicts.case_dict_names()
    case_dict, num_points, num_skip_rows = db2case[database_name]
    database = np.loadtxt(database_name, skiprows=num_skip_rows)

    # Split database according to zonal boundary marker
    incl_zonal_markers = True
    num_zonal_markers = 1

    # Var  Cx | Cy | k | eps | gradU | gradp |  U  | gradk |   marker inputs   |  zonal marker  |tauij|
    # Col   0    1   2    3     4-12   13-15  16-18  19-21   -(nm+nz+9):-(nz+9)    -(nz+9):-9     -9:

    zonal_mark_col = database[:, -(num_zonal_markers+9):-9]
    z1_to_db_idx = []
    z2_to_db_idx = []
    mod_inst = 0
    assert num_zonal_markers == 1, "more than one zonal marker used, code needs updating"

    for i in range(zonal_mark_col.shape[0]):
        # Obtain row idx for each zone
        if zonal_mark_col[i] < 0.25:  # Else data
            z1_to_db_idx.append(i)
        else:  # Separation data
            z2_to_db_idx.append(i)

        # Obtain list containing num of points per case for each zone
        if (i+1) % num_points == 0:
            mod_inst += 1
            if mod_inst == 1:
                z1_num_points = [len(z1_to_db_idx)]
                z2_num_points = [len(z2_to_db_idx)]
            else:
                z1_num_points.append(len(z1_to_db_idx) - z1_num_points[mod_inst-1])
                z2_num_points.append(len(z2_to_db_idx) - z2_num_points[mod_inst-1])

    # Construct database for each zone using their row idx
    z1_database = database[z1_to_db_idx, :]
    z2_database = database[z2_to_db_idx, :]
    assert sum(z1_num_points) == z1_database.shape[0], "mismatch between num points list and num of database entries"
    assert sum(z2_num_points) == z2_database.shape[0], "mismatch between num points list and num of database entries"

    # Construct dictionary for mapping cases to num of points in each zone
    def zonal_case_dict(case_dict, num_points_list):
        dict = case_dict
        for count, value in enumerate(num_points_list):
            dict[count] = value
        return dict

    z1_case_dict = zonal_case_dict(case_dict, z1_num_points)
    z2_case_dict = zonal_case_dict(case_dict, z2_num_points)

    # Run TBNN for zone 1
    start = timeit.default_timer()
    folder_path, current_folder = frontend.tbnn_main(z1_database, z1_case_dict, incl_zonal_markers=incl_zonal_markers,
                                                     num_zonal_markers=num_zonal_markers)
    stop = timeit.default_timer()
    results_writer.write_time(start, stop, folder_path, current_folder)

    # Run TBNN for zone 2
    start = timeit.default_timer()
    folder_path, current_folder = frontend.tbnn_main(z2_database, z2_case_dict, incl_zonal_markers=incl_zonal_markers,
                                                     num_zonal_markers=num_zonal_markers)
    stop = timeit.default_timer()
    results_writer.write_time(start, stop, folder_path, current_folder)

    # Run TKENN for zone 1
    start = timeit.default_timer()
    folder_path, current_folder = tke_frontend.tkenn_main(z1_database, z1_case_dict,
                                                          incl_zonal_markers=incl_zonal_markers,
                                                          num_zonal_markers=num_zonal_markers)
    stop = timeit.default_timer()
    results_writer.write_time(start, stop, folder_path, current_folder)

    # Run TKENN for zone 2
    start = timeit.default_timer()
    folder_path, current_folder = tke_frontend.tkenn_main(z2_database, z2_case_dict,
                                                          incl_zonal_markers=incl_zonal_markers,
                                                          num_zonal_markers=num_zonal_markers)
    stop = timeit.default_timer()
    results_writer.write_time(start, stop, folder_path, current_folder)


if __name__ == "__main__":
    zonal_driver()
    print("Zonal driver finished")
