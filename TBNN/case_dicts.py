def get_metadata(dataset_name):
    ref_dict = {"CHAN7_dataset.txt": chan7(),
                "FBFS5_dataset.txt": fbfs5(),
                "FBFS5_IMPJ20000_dataset.txt": fbfs5_impj20000(),
                "PHLL4_dataset.txt": phll4(),
                "CHAN7_no_oti_half_dataset.txt": half_chan7_no_oti(),
                "CHAN7_with_interps_half_dataset.txt": half_chan7_with_interps(),
                "CHAN7_no_oti_with_interps_half_dataset.txt":
                    half_chan7_no_oti_with_interps(),
                "MVEN6_dataset.txt": mven6(),
                "DUCT16_dataset.txt": duct16()}
    return ref_dict[dataset_name]


def chan7():
    keys = [180, 290, 395, 490, 590, 760, 945]
    num_points = 182
    case_dict = dict.fromkeys(keys, num_points)
    num_skip_rows = 1
    return case_dict, num_points, num_skip_rows


def fbfs5():  # ✓
    keys = [1, 2, 2.5, 3, 4]
    num_points = 80296
    case_dict = dict.fromkeys(keys, num_points)
    num_skip_rows = 1
    return case_dict, num_points, num_skip_rows


def fbfs5_impj20000():
    keys = [1, 2, 2.5, 3, 4, 20000]
    num_points = [80296, 80296, 80296, 80296, 80296, 112006]
    case_dict = dict(zip(keys, num_points))
    num_skip_rows = 0
    return case_dict, num_points, num_skip_rows


def phll4():
    keys = [0.5, 1, 1.2, 1.5]
    num_points = 14751
    case_dict = dict.fromkeys(keys, num_points)
    num_skip_rows = 1
    return case_dict, num_points, num_skip_rows


def half_chan7_no_oti():
    keys = [180, 290, 395, 490, 590, 760, 945]
    num_points = [56, 56, 55, 54, 54, 53, 53]
    case_dict = dict(zip(keys, num_points))
    num_skip_rows = 1
    return case_dict, num_points, num_skip_rows


def half_chan7_with_interps():
    keys = [180, 290, 395, 490, 590, 760, 945]
    num_points = 600
    case_dict = dict.fromkeys(keys, num_points)
    num_skip_rows = 1
    return case_dict, num_points, num_skip_rows


def half_chan7_no_oti_with_interps():
    keys = [180, 290, 395, 490, 590, 760, 945]
    num_points = 400
    case_dict = dict.fromkeys(keys, num_points)
    num_skip_rows = 1
    return case_dict, num_points, num_skip_rows


def mven6():
    keys = [0.5, 1.0, 1.2, 1.5, 3500, 3600]
    num_points = [14751, 14751, 14751, 14751, 9216, 80296]
    case_dict = dict(zip(keys, num_points))
    num_skip_rows = 1
    return case_dict, num_points, num_skip_rows


def duct16():
    keys = [1100, 1150, 1250, 1300, 1350, 1400, 1500, 1600, 1800, 2000, 2205, 2400,
            2600, 2900, 3200, 3500]
    num_points = 9216
    case_dict = dict.fromkeys(keys, num_points)
    num_skip_rows = 1
    return case_dict, num_points, num_skip_rows
