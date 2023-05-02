"""
This script executes the splitting of data in accordance with zones for TBNN and TKENN
use. The splitting is performed separately for pre-declared training, validation and
testing case lists. This script was written for the Apr 2023 work by Anthony Man at
The University of Manchester.

Zonal markers calculated for Apr 2023 checked with plots.
"""

import numpy as np
import sys
from zonal_core import create_zonal_db, create_non_zonal_db, shuffle_rows


def main():
    sys.path.insert(1, "../TBNN")
    sys.path.insert(2, "../TKE_NN")

    from TBNN import frontend
    from TKE_NN import tke_frontend

    train_list = ["BUMP_h20", "BUMP_h38", "FBFS_1800"]
    valid_list = ["BUMP_h26"]
    test_list = ["FBFS_5400"]
    coords_list = ["Cx", "Cy"]
    marker_list = ["nd_Q", "nd_TI", "nd_Ux", "Re_y"]
    zones = marker_list + ["fstrm"]
    num_cols = 22  # Cx, Cy, k, epsilon, gradU(9), true_tauij(9)

    # Prepare zonal training, validation and testing data dictionaries
    zonal_train_dataset_dict = create_zonal_db(train_list, coords_list, marker_list,
                                               zones, num_cols)
    zonal_train_dataset_dict = shuffle_rows(zonal_train_dataset_dict, seed=1)
    zonal_valid_dataset_dict = create_zonal_db(valid_list, coords_list, marker_list,
                                               zones, num_cols)
    zonal_test_dataset_dict = create_zonal_db(test_list, coords_list, marker_list,
                                              zones, num_cols)

    # Run TBNN and TKENN for each zone
    # frontend.tbnn_main(np.nan, np.nan, zones=zones,
    #                    zonal_train_dataset=zonal_train_dataset_dict,
    #                    zonal_valid_dataset=zonal_valid_dataset_dict,
    #                    zonal_test_dataset=zonal_test_dataset_dict, version="v2")
    tke_frontend.tkenn_main(np.nan, np.nan, zones=zones,
                            zonal_train_dataset=zonal_train_dataset_dict,
                            zonal_valid_dataset=zonal_valid_dataset_dict,
                            zonal_test_dataset=zonal_test_dataset_dict, version="v2")

    # Prepare non-zonal training, validation and testing datasets
    train_dataset_dict = create_non_zonal_db(train_list)
    train_dataset_dict = shuffle_rows(train_dataset_dict, seed=1)
    valid_dataset_dict = create_non_zonal_db(valid_list)
    test_dataset_dict = create_non_zonal_db(test_list)

    # Run non-zonal TBNN and TKENN to compare against zonal results
    frontend.tbnn_main(np.nan, np.nan, zones=["non_zonal"],
                       zonal_train_dataset=train_dataset_dict,
                       zonal_valid_dataset=valid_dataset_dict,
                       zonal_test_dataset=test_dataset_dict, version="v2")
    tke_frontend.tkenn_main(np.nan, np.nan, zones=["non_zonal"],
                            zonal_train_dataset=train_dataset_dict,
                            zonal_valid_dataset=valid_dataset_dict,
                            zonal_test_dataset=test_dataset_dict, version="v2")


if __name__ == "__main__":
    main()
    print("Zonal driver finished")
