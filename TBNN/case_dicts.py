def case_dict_names():
    database2case = {
        "CHAN7_database.txt": channel_flow_7(),
        "FBFS_full_set_no_walls.txt": fbfs_3(),
        "FBFS5_full_set_no_walls.txt": fbfs_5(),
        "FBFS5_IMPJ20000_dataset.txt": fbfs_5_impj_20000(),
        "PHLL4_dataset.txt": phll_4()
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


def fbfs_5_impj_20000():
    keys = [1, 2, 2.5, 3, 4, 20000]
    num_points = [80296, 80296, 80296, 80296, 80296, 112006]
    case_dict = dict(zip(keys, num_points))
    num_skip_rows = 0
    return case_dict, num_points, num_skip_rows


def phll_4():
    keys = [0.5, 1, 1.2, 1.5]
    num_points = 14751
    case_dict = dict.fromkeys(keys, num_points)
    num_skip_rows = 1
    return case_dict, num_points, num_skip_rows
