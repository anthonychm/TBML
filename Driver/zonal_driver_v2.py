"""
This script executes the splitting of data in accordance with zones for TBNN and TKENN
use. The splitting is performed separately for pre-declared training, validation and
testing case lists. This script was written for the Apr 2023 work by Anthony Man at
The University of Manchester.

Zonal markers calculated for Apr 2023 checked with plots.
Code checked and debugged 06/05/2023
"""

import numpy as np
import sys
from zonal_core import create_zonal_db, create_non_zonal_db, shuffle_rows


def main():
    sys.path.insert(1, "../TBNN")
    sys.path.insert(2, "../TKE_NN")

    from TBNN import frontend
    from TKE_NN import tke_frontend

    # Specify zonal variables ✓
    train_list = ["BUMP_h20", "BUMP_h38", "FBFS_1800"]
    valid_list = ["BUMP_h26"]
    test_list = ["FBFS_5400"]
    coords_list = ["Cx", "Cy"]
    marker_list = ["nd_Q", "nd_TI", "nd_Ux", "Re_y"]
    zones = marker_list + ["fstrm"]
    num_cols = 22  # Cx, Cy, k, epsilon, gradU(9), true_tauij(9)

    # Prepare zonal training, validation and testing dictionaries ✓
    ztrain_db_dict, _, _ = \
        create_zonal_db(train_list, coords_list, marker_list, zones, num_cols)  # ✓
    ztrain_db_dict = shuffle_rows(ztrain_db_dict, seed=1)  # ✓
    zvalid_db_dict, _, _ = \
        create_zonal_db(valid_list, coords_list, marker_list, zones, num_cols)  # ✓
    ztest_db_dict, ztest_case_tag_dict, ztest_k_normzr_dict = \
        create_zonal_db(test_list, coords_list, marker_list, zones, num_cols)  # ✓

    # Run TBNN and TKENN for each zone ✓
    frontend.tbnn_main(np.nan, np.nan, zones=zones, zonal_train_dataset=ztrain_db_dict,
                       zonal_valid_dataset=zvalid_db_dict,
                       zonal_test_dataset=ztest_db_dict, version="v2")  # ✓
    tke_frontend.\
        tkenn_main(np.nan, np.nan, zones=zones, zonal_train_dataset=ztrain_db_dict,
                   zonal_valid_dataset=zvalid_db_dict, zonal_test_dataset=ztest_db_dict,
                   zonal_test_case_tags=ztest_case_tag_dict,
                   zonal_test_output_normzr=ztest_k_normzr_dict, version="v2")  # ✓

    # Prepare non-zonal training, validation and testing datasets ✓
    train_db_dict, _, _ = create_non_zonal_db(train_list)  # ✓
    train_db_dict = shuffle_rows(train_db_dict, seed=1)  # ✓
    valid_db_dict, _, _ = create_non_zonal_db(valid_list)  # ✓
    test_db_dict, case_tag_dict, test_k_normzr_dict = create_non_zonal_db(test_list)  # ✓

    # Run non-zonal TBNN and TKENN to compare against zonal results ✓
    frontend.tbnn_main(np.nan, np.nan, zones=["non_zonal"],
                       zonal_train_dataset=train_db_dict,
                       zonal_valid_dataset=valid_db_dict,
                       zonal_test_dataset=test_db_dict, version="v2")  # ✓
    tke_frontend.\
        tkenn_main(np.nan, np.nan, zones=["non_zonal"], zonal_train_dataset=train_db_dict,
                   zonal_valid_dataset=valid_db_dict, zonal_test_dataset=test_db_dict,
                   zonal_test_case_tags=case_tag_dict,
                   zonal_test_output_normzr=test_k_normzr_dict, version="v2")  # ✓


if __name__ == "__main__":
    main()
    print("Zonal driver finished")
