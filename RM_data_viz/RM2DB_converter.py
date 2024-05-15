"""
This script converts RM case data to the database format used in AM's TBNN and TBMix.
Checked 14/05/2024 [✓]
"""

import numpy as np

# User defined settings
db_name = "PHLL5"  # Name of the database
turb_model = "komegasst"  # Name of turbulence model
case_list = ["PHLL_case_0p5", "PHLL_case_0p8", "PHLL_case_1p0", "PHLL_case_1p2",
             "PHLL_case_1p5"]  # Cases to include in database
rans_list = ["Cx", "Cy", "k", "epsilon", "gradU"]
zonal_list = []
labels_list = ["tau"]
num_cols_dict = {"Cx": 1, "Cy": 1, "k": 1, "epsilon": 1, "gradU": 9, "tau": 9}

parent_path = ("C:/Users/Antho/Dropbox (The University of Manchester)/PhD_Anthony_Man/"
               "AM_PhD_shared_documents/McConkey data")
vars_list = rans_list + zonal_list + labels_list
list_path_dict = {"rans": parent_path + '/' + turb_model + '/' + turb_model + '_',
                  "zonal": parent_path + '/' + turb_model + '/zonal criteria/',
                  "labels": parent_path + '/labels/'}

# Initialise database array
num_rows, num_cols = 0, 0
num_rows_dict = {}

for case in case_list:
    data = np.load(list_path_dict["rans"] + case + '_Cx.npy')
    num_rows_dict[case] = len(data)
    num_rows += len(data)
for var in vars_list:
    num_cols += num_cols_dict[var]
db = np.full((num_rows, num_cols), np.nan)

# Loop through case_list and variables to fill database
row_count = 0
for case in case_list:
    col_count = 0
    for var in vars_list:
        # Load variable data
        if var in rans_list:
            path = list_path_dict["rans"]
        elif var in zonal_list:
            path = list_path_dict["zonal"]
        elif var in labels_list:
            path = list_path_dict["labels"]
        else:
            raise Exception("Variable not present in variable list")

        try:
            data = np.load(path + case + "_" + var + '.npy')
        except:
            data = np.loadtxt(path + case + "_" + var + '.txt')

        # Update database with variable data
        db[row_count:row_count + num_rows_dict[case],
            col_count:col_count + num_cols_dict[var]] = data
        col_count += num_cols_dict[var]
    row_count += num_rows_dict[case]

# Write database to file
assert np.any(np.isnan(db)) is False
np.savetxt(db_name + "_dataset.txt", db, delimiter=' ',
           header=("Cx" "Cy" "tke" "epsilon" "dU/dx" "dU/dy" "dU/dz" "dV/dx" "dV/dy" 
                   "dV/dz" "dW/dx" "dW/dy" "dW/dz" "uu_11" "uu_12" "uu_13" "uu_21" 
                   "uu_22" "uu_23" "uu_31" "uu_32" "uu_33"))

# orig label x, y, z match rans Cx, Cy and Cz?
