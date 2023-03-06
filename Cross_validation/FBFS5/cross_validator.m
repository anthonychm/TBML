%% PREAMBLE %%

% This script checks hyperparameter tuning results and performs
% cross-validation 

%% Create cv_class object and load results

clear variables
addpath('..\')
cv_obj = cv_class;
cv_obj.model_type = 'TKENN';
cv_obj.combs = [5, 2, 3, 3, 4, 4];
cv_obj.num_rows_target = 1440;
[num_rows, hpt_results] = read_hpt_results(cv_obj);

%% Run hyperparameter tuning results checks

% Make num_layers_dict
keys = [1, 2, 3, 4];
vals = [2, 5, 10, 20];
[num_layers_dict, num_runs_per_layers] = ...
    cv_obj.make_num_layers_dict(keys, vals, num_rows);

% Make batch_size_dict
batch_size_dict = cv_obj.make_batch_size_dict();

% Define turnover values
af_turnover = 90;
tvt_turnover = 30;
gamma_turnover = 10;

% Run assert checks
for i = 1:num_rows
    cv_obj.check_run_idx(i, hpt_results)
    cv_obj.check_job_idx(i, hpt_results)
    cv_obj.check_trial_idx(i, hpt_results)
    cv_obj.check_num_layers(i, num_runs_per_layers, hpt_results, ...
        num_layers_dict)
    num_nodes = cv_obj.rm_list_symbols(i, hpt_results, 'num_hid_nodes');
    cv_obj.check_num_nodes(i, num_nodes, hpt_results, num_runs_per_layers)
    af = cv_obj.rm_list_symbols(i, hpt_results, 'af');
    cv_obj.check_af(i, af, hpt_results, af_turnover);
    cv_obj.check_tvt_lists(i, tvt_turnover, hpt_results);
    cv_obj.check_batch_size(i, hpt_results, batch_size_dict)
    cv_obj.check_gamma(i, hpt_results, gamma_turnover)
end

%% Perform cross-validation

% Define constants
num_folds = 3;
q = tvt_turnover/num_folds;
q_count = 0;
run_cv = true;

% Perform cross-validation
for i = 1:num_rows
    [run_cv, q_count] = cv_obj.determine_run_cv(i, q, q_count, run_cv);
    
    if run_cv == true
        new_row = cv_obj.init_new_cv_row(i, hpt_results);
        cv_obj.check_hp_match(i, new_row, hpt_results, q)
        new_row{7} = cv_obj.threef_cv_stat(i, q, hpt_results, 'run_time');
        new_row{8} = cv_obj.threef_cv_stat(i, q, hpt_results, ...
            'Mean_final_training_rmse');
        new_row{9} = cv_obj.threef_cv_stat(i, q, hpt_results, ...
            'Mean_final_validation_rmse');
        new_row{10} = cv_obj.threef_cv_stat(i, q, hpt_results, ...
            'Mean_testing_rmse');
        
        if i == 1
            cv_results = new_row;
        else
            cv_results = [cv_results; new_row];
        end
    end
end


%% Post-cross-validation

% Check cross-validation results
cv_obj.check_cv_results(q_count, num_rows, cv_results)

% Sort cross-validation results
sort_crit = 'valid_error';
cv_results_sorted = cv_obj.sort_cv_results(cv_results, sort_crit);

% Write cross-validation results
% write_cv_results(cv_obj, cv_results)
