def case_dict_names():
    database2case = {
        "full_set_no_walls.txt": channel_flow_7(),
        "FBFS_full_set_no_walls.txt": fbfs_3(),
        "FBFS_full_set_no_walls_5.txt": fbfs_5()
    }
    return database2case


def channel_flow_7():
    case_dict = {180: 1, 290: 2, 395: 3, 490: 4, 590: 5, 760: 6, 945: 7}
    n_case_points = 182
    return case_dict, n_case_points


def fbfs_3():
    case_dict = {1: 1, 2: 2, 4: 3}
    n_case_points = 80296
    return case_dict, n_case_points


def fbfs_5():
    case_dict = {1: 1, 2: 2, 2.5: 3, 3: 4, 4: 5}
    n_case_points = 80296
    return case_dict, n_case_points
