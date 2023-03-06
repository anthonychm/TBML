def case_dict_names():
    database2case = {
        "full_set_no_walls.txt": channel_flow_7(),
        "FBFS_full_set_no_walls.txt": fbfs_3(),
        "FBFS5_full_set_no_walls.txt": fbfs_5()
    }
    return database2case


def channel_flow_7():
    keys = [180, 290, 395, 490, 590, 760, 945]
    num_points = 182
    case_dict = dict.fromkeys(keys, num_points)
    num_skip_rows = 1
    return case_dict, num_points, num_skip_rows


def fbfs_3():
    keys = [1, 2, 4]
    num_points = 80296
    case_dict = dict.fromkeys(keys, num_points)
    num_skip_rows = 1
    return case_dict, num_points, num_skip_rows


def fbfs_5():  # ✓
    keys = [1, 2, 2.5, 3, 4]
    num_points = 80296
    case_dict = dict.fromkeys(keys, num_points)
    num_skip_rows = 1
    return case_dict, num_points, num_skip_rows
