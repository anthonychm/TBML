classdef cv_class
    % This class contains methods for running pre-cross-validation checks
    % and performing cross-validation
    
    properties
        model_type {mustBeText} = 'dummy'
        combs {mustBeVector} = [0, 0]
        num_rows_target {mustBeInteger} = 0
    end
    
    methods
        function [num_rows, hpt_results] = read_hpt_results(obj)
            % Read hyperparameter tuning results
            num_rows = 1;
            for i = 1:length(obj.combs)
                num_rows = num_rows*obj.combs(i);
            end
            assert(num_rows == obj.num_rows_target)
            hpt_results = readtable(strcat(obj.model_type, ...
                '_hp_tuning_all_results.csv'));
            assert(size(hpt_results, 1) == num_rows)
        end
        
        function write_cv_results(obj, cv_results)
            % Write cv_results to file
            header = {'run', 'num_layers', 'num_nodes', 'afs', ...
                'lr_scheduler_params', 'batch_size', 'run_time', ...
                'train_error', 'valid_error', 'test_error'};
            cv_table = cell2table(cv_results, 'VariableNames', header);
            writetable(cv_table, strcat(obj.model_type, '_cv_results.csv'))
        end
    end
    
    methods (Static)
        function check_run_idx(i, hpt_results)
            % Check run indexes
            assert(hpt_results{i, 'run'} == i)
        end
        
        function check_job_idx(i, hpt_results)
            % Check job indexes
            if rem(i/10, 1) == 0
                assert(hpt_results{i, 'job'} == floor(i/10))
            else
                assert(hpt_results{i, 'job'} == floor(i/10) + 1)
            end
        end
        
        function check_trial_idx(i, hpt_results)
            % Check trial indexes
            if rem(i, 10) == 0
                assert(hpt_results{i, 'trial'} == 10)
            else
                assert(hpt_results{i, 'trial'} == rem(i, 10))
            end
        end
        
        function [num_layers_dict, num_runs_per_layers] = ...
                make_num_layers_dict(keys, vals, num_rows)
            % Make num layers dictionary
            num_layers_dict = containers.Map(keys, vals);
            num_runs_per_layers = num_rows/length(keys);
        end
        
        function check_num_layers(i, num_runs_per_layers, hpt_results, ...
                num_layers_dict)
            % Check number of hidden layers
            if rem(i/num_runs_per_layers, 1) == 0
                lower_bound = floor(i/num_runs_per_layers);
            else
                lower_bound = floor(i/num_runs_per_layers) + 1;
            end
            assert(hpt_results{i, 'num_hid_layers'} == ...
                num_layers_dict(lower_bound))
        end
        
        function param = rm_list_symbols(i, hpt_results, param_string)
            % Remove , [ and ] from lists
            param = cell2mat(hpt_results{i, param_string});
            param = strsplit(param, ', ');
            param = erase(param, '[');
            param = erase(param, ']');
        end
        
        function check_num_nodes(i, num_nodes, hpt_results, ...
                num_runs_per_layers)
            % Check number of hidden nodes
            assert(length(num_nodes) == hpt_results{i, 'num_hid_layers'})
            nodes_turnover = num_runs_per_layers;
            if rem(i, nodes_turnover) ~= 0 && ...
                    rem(i, nodes_turnover) <= (0.25*nodes_turnover)
                assert(str2double(unique(num_nodes)) == 5)
            elseif rem(i, nodes_turnover) > (0.25*nodes_turnover) && ...
                    rem(i, nodes_turnover) <= (0.5*nodes_turnover)
                assert(str2double(unique(num_nodes)) == 10)
            elseif rem(i, nodes_turnover) > (0.5*nodes_turnover) && ...
                    rem(i, nodes_turnover) <= (0.75*nodes_turnover)
                assert(str2double(unique(num_nodes)) == 25)
            else
                assert(str2double(unique(num_nodes)) == 50)
            end
        end
        
        function check_af(i, af, hpt_results, af_turnover)
            % Check activation functions and weight initializer parameters
            [idx, af_types] = findgroups(af);
            af_counts = histcounts(idx);
            weight_init_params = cell2mat(hpt_results{i, ...
                'weight_init_params'});

            assert(af_counts == hpt_results{i, 'num_hid_layers'})
            assert(length(af_types) == 1)

            if rem(i, af_turnover) ~= 0 && ...
                    rem(i, af_turnover) <= ((1/3)*af_turnover)
                assert(strcmp(af_types, "'ReLU'") == 1)
                assert(strcmp(weight_init_params, ...
                    "nonlinearity=relu") == 1)
            elseif rem(i, af_turnover) > ((1/3)*af_turnover) && ...
                    rem(i, af_turnover) <= ((2/3)*af_turnover)
                assert(strcmp(af_types, "'ELU'") == 1)
                assert(strcmp(weight_init_params, ...
                    "nonlinearity=leaky_relu") == 1)
            else
                assert(strcmp(af_types, "'SiLU'") == 1)
                assert(strcmp(weight_init_params, ...
                    "nonlinearity=leaky_relu") == 1)
            end
        end
        
        function check_tvt_lists(i, tvt_turnover, hpt_results)
            % Check training, validation and testing lists
            if rem(i, tvt_turnover) ~= 0 && ...
                    rem(i, tvt_turnover) <= (1/3)*tvt_turnover
                assert(strcmp(hpt_results{i, 'train_list'}, ...
                    '[1, 2.5]') == 1)
                assert(strcmp(hpt_results{i, 'valid_list'}, '[3]') == 1)
            elseif rem(i, tvt_turnover) > ((1/3)*tvt_turnover) && ...
                    rem(i, tvt_turnover) <= ((2/3)*tvt_turnover)
                assert(strcmp(hpt_results{i, 'train_list'}, '[1, 3]') == 1)
                assert(strcmp(hpt_results{i, 'valid_list'}, '[2.5]') == 1)
            else
                assert(strcmp(hpt_results{i, 'train_list'}, ...
                    '[2.5, 3]') == 1)
                assert(strcmp(hpt_results{i, 'valid_list'}, '[1]') == 1)
            end
            assert(strcmp(hpt_results{i, 'test_list'}, '[2, 4]') == 1)
        end
        
        function batch_size_dict = make_batch_size_dict()
            % Make batch size dictionary
            keys = [1, 2, 3, 4, 5, 0];
            values = [16, 32, 64, 128, 256, 256];
            batch_size_dict = containers.Map(keys, values);
        end
        
        function check_batch_size(i, hpt_results, batch_size_dict)
            % Check batch sizes
            if i < 6
                assert(hpt_results{i, 'batch_size'} == batch_size_dict(i))
            else
                assert(hpt_results{i, 'batch_size'} == ...
                    batch_size_dict(rem(i, 5)))
            end
        end
        
        function check_gamma(i, hpt_results, gamma_turnover)
            % Check exponential learning rate scheduler gamma
            lrs_params = cell2mat(hpt_results{i, 'lr_scheduler_params'});
            lrs_params = strsplit(lrs_params, '='); 
            gamma = str2double(lrs_params{2});
            assert(length(lrs_params) == 2)

            if rem(i, 10) ~= 0 && rem(i, 10) <= (0.5*gamma_turnover)
                assert(gamma == 0.995)
            else
                assert(gamma == 0.98)
            end
        end
        
        function [run_cv, q_count] = determine_run_cv(i, q, q_count, run_cv)
            % Determine whether to run cross-validation for current row
            % q = num of runs per interval of tvt combination
            % (q = tvt_turnover/num_folds)
            if i > 1 && rem(i, q) == 1
                q_count = q_count + 1;
                run_cv = false;
                if rem(q_count, 3) == 0
                    run_cv = true;
                end
            end
        end
        
        function new_row = init_new_cv_row(i, hpt_results)
            % Initialise new cross-validation results row with
            % hyperparameter information
            new_row = cell(1, 10);
            new_row{1} = hpt_results{i, 'run'};
            new_row{2} = hpt_results{i, 'num_hid_layers'};
            new_row{3} = hpt_results{i, 'num_hid_nodes'};
            new_row{4} = hpt_results{i, 'af'};
            new_row{5} = hpt_results{i, 'lr_scheduler_params'};
            new_row{6} = hpt_results{i, 'batch_size'};
        end
        
        function check_hp_match(i, new_row, hpt_results, q)
            % Check hyperparameter information matches with other data to
            % be cross-validated
            assert(new_row{2} == hpt_results{i+q, 'num_hid_layers'})
            assert(new_row{2} == hpt_results{i+(2*q), 'num_hid_layers'})
            assert(strcmp(new_row{3}, ...
                hpt_results{i+q, 'num_hid_nodes'}) == 1)
            assert(strcmp(new_row{3}, ...
                hpt_results{i+(2*q), 'num_hid_nodes'}) == 1)
            assert(strcmp(new_row{4}, hpt_results{i+q, 'af'}) == 1)
            assert(strcmp(new_row{4}, hpt_results{i+(2*q), 'af'}) == 1)
            assert(strcmp(new_row{5}, ...
                hpt_results{i+q, 'lr_scheduler_params'}) == 1)
            assert(strcmp(new_row{5}, ...
                hpt_results{i+(2*q), 'lr_scheduler_params'}) == 1)
            assert(new_row{6} == hpt_results{i+q, 'batch_size'})
            assert(new_row{6} == hpt_results{i+(2*q), 'batch_size'})
        end
        
        function cv_stat = threef_cv_stat(i, q, hpt_results, stat_name)
            % Calculate cross-validated statistic by averaging over the 3
            % folds
            cv_stat = mean([hpt_results{i, stat_name}, ...
                hpt_results{i+q, stat_name}, ...
                hpt_results{i+(2*q), stat_name}]);
        end
        
        function check_cv_results(q_count, num_rows, cv_results)
            % Check cv_results cell array
            assert(q_count == (num_rows/10) - 1)
            assert(size(cv_results, 1) == num_rows/3)
            assert(size(cv_results, 2) == 10)
        end
        
        function cv_results_sorted = sort_cv_results(cv_results, sort_crit)
            % Sort cv_results according to sorting criteria
            keys = {'runtime', 'train_error', 'valid_error', 'test_error'};
            sort_dict = containers.Map(keys, 7:10);
            cv_results_sorted = sortrows(cv_results, sort_dict(sort_crit));
        end
    end
end

