import pandas as pd
import os

current_dir = os.getcwd()
results_all = pd.read_csv(os.path.join(current_dir, 'TKENN_hp_tuning_all_results.csv'))

for i in range(1, 145):
    job_csv = pd.read_csv(os.path.join(current_dir, 'Job' + str(i), 'TBNN output data',
                                       'Results', 'Trial_parameters_and_means.csv'))
    results_all = pd.concat([results_all, job_csv])
results_all.to_csv(os.path.join(current_dir, 'TKENN_hp_tuning_all_results.csv'),
                   index=False)
