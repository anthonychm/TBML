import numpy as np
import os
import sys


def zonal_driver():
    sys.path.insert(1, "../TBNN")
    sys.path.insert(2, "../TKE_NN")

    from TBNN import frontend, case_dicts
    from TKE_NN import tke_frontend

    # Load database and associated dictionary ✓
    database_name = "FBFS5_full_set_no_walls.txt"
    db2case = case_dicts.case_dict_names()
    case_dict, num_points, num_skip_rows = db2case[database_name]  # ✓
    database = np.loadtxt(database_name, skiprows=num_skip_rows)
    print('Full database loaded')

    # Split database according to zonal boundary marker ✓
    incl_zonal_markers = True
    num_zonal_markers = 1

    # For 2D:
    # Var  Cx | Cy | k | eps | gradU | gradp |  U  | gradk |   marker inputs   |
    # Col   0    1   2    3     4-12   13-15  16-18  19-21   -(nm+nz+9):-(nz+9)

    # Var |  zonal marker  |tauij|
    # Col     -(nz+9):-9     -9:

    zonal_mark_col = database[:, -(num_zonal_markers+9):-9]
    z1_to_db_idx = []
    z2_to_db_idx = []
    mod_inst = 0
    assert num_zonal_markers == 1, "more than one zonal marker used, code needs updating"

    for i in range(zonal_mark_col.shape[0]):
        # Obtain row idx for each zone ✓
        if zonal_mark_col[i] < 0.1:  # Else data
            z1_to_db_idx.append(i)
        else:  # Separation data
            z2_to_db_idx.append(i)

        # Obtain list containing num of points per case for each zone ✓
        if (i+1) % num_points == 0:
            mod_inst += 1
            if mod_inst == 1:
                z1_num_points = [len(z1_to_db_idx)]
                z2_num_points = [len(z2_to_db_idx)]
            else:
                z1_num_points.append(len(z1_to_db_idx) - sum(z1_num_points))
                z2_num_points.append(len(z2_to_db_idx) - sum(z2_num_points))

    # Construct database for each zone using their row idx ✓
    z1_database = database[z1_to_db_idx, :]
    z2_database = database[z2_to_db_idx, :]
    assert sum(z1_num_points) == z1_database.shape[0], \
        "mismatch between num points list and num of database entries"
    assert sum(z2_num_points) == z2_database.shape[0], \
        "mismatch between num points list and num of database entries"
    print('Zonal databases created')

    # Construct dictionary for mapping cases to num of points in each zone ✓
    def zonal_case_dict(case_dict, num_points_list):
        case_dict_cp = case_dict.copy()
        counter = 0
        for key in case_dict_cp:
            case_dict_cp[key] = num_points_list[counter]
            counter = counter + 1
        return case_dict_cp

    z1_case_dict = zonal_case_dict(case_dict, z1_num_points)  # ✓
    z2_case_dict = zonal_case_dict(case_dict, z2_num_points)  # ✓
    print('Zonal case dictionaries created')

    # Run TBNN for zone 1 ✓
    #frontend.tbnn_main(z1_database, z1_case_dict, incl_zonal_markers=incl_zonal_markers,
                       #num_zonal_markers=num_zonal_markers)  # ✓
    #os.rename(os.path.join(os.getcwd(), "TBNN output data"),
    #          os.path.join(os.getcwd(), "Zone 1 TBNN output data"))

    # Run TBNN for zone 2 ✓
    #frontend.tbnn_main(z2_database, z2_case_dict, incl_zonal_markers=incl_zonal_markers,
                       #num_zonal_markers=num_zonal_markers)  # ✓
    #os.rename(os.path.join(os.getcwd(), "TBNN output data"),
    #          os.path.join(os.getcwd(), "Zone 2 TBNN output data"))

    # Run TKENN for zone 1 ✓
    #tke_frontend.tkenn_main(z1_database, z1_case_dict,
                            #incl_zonal_markers=incl_zonal_markers,
                            #num_zonal_markers=num_zonal_markers)  # ✓
    #os.rename(os.path.join(os.getcwd(), "TBNN output data"),
    #          os.path.join(os.getcwd(), "Zone 1 TKENN output data"))

    # Run TKENN for zone 2 ✓
    tke_frontend.tkenn_main(z2_database, z2_case_dict,
                            incl_zonal_markers=incl_zonal_markers,
                            num_zonal_markers=num_zonal_markers)  # ✓
    #os.rename(os.path.join(os.getcwd(), "TBNN output data"),
    #          os.path.join(os.getcwd(), "Zone 2 TKENN output data"))


if __name__ == "__main__":
    zonal_driver()
    print("Zonal driver finished")
