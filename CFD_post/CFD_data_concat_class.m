classdef CFD_data_concat_class
    % An instance of this class is a CFD data concatenator for writing a
    % CFD data file to be read by TBNN python code
    
    properties
        param_values {mustBeVector} = NaN(1, 2)
        case_prefix {mustBeText} = 'dummy'
        in_table_name {mustBeText} = 'dummy'
        num_RANS_coords {mustBeInteger} = 0
        num_RANS_cols {mustBeInteger} = 0
        num_LES_coords {mustBeInteger} = 0
        num_LES_cols {mustBeInteger} = 0
        num_cases {mustBeInteger} = 0
        num_dims {mustBeInteger} = 0
        num_at_walls {mustBeInteger} = 0
    end
    
    methods
        function [RANS_data, LES_data] = read_CFD_data(obj, RANS_data, ...
                LES_data) % ✓
            % Read RANS and LES data for a set of CFD cases
            
            counter = 0;
            for param = obj.param_values
                counter = counter + 1;
                case_name = strcat(obj.case_prefix, num2str(param));
                
                % Load RANS data (excl. tauij) ✓
                RANS_table = load(strcat('Results\', case_name, ...
                    '_RANS_', obj.in_table_name, '.mat'));
                RANS_data(:, :, counter) = table2array(...
                    RANS_table.table(:, 1:end - 6));
                RANS_slice = RANS_data(:, :, counter);
                
                % Load LES data ✓
                LES_table = load(strcat('Results\', case_name, ...
                    '_LES_', obj.in_table_name, '.mat'));
                LES_data(:, :, counter) = table2array(LES_table.table);
                LES_slice = LES_data(:, :, counter);
            end
        end
        
        function init_assert_checks(obj, RANS_data, LES_data) % ✓
            % Perform initial assert checks on RANS_data and LES_data
            
            assert(sum(isnan(RANS_data), 'all') == 0)
            assert(sum(isnan(LES_data), 'all') == 0)
            
            assert(size(RANS_data, 1) == obj.num_RANS_coords)
            assert(size(RANS_data, 2) == obj.num_RANS_cols)
            assert(size(RANS_data, 3) == obj.num_cases)

            assert(size(LES_data, 1) == obj.num_LES_coords)
            assert(size(LES_data, 2) == obj.num_LES_cols)
            assert(size(LES_data, 3) == obj.num_cases)
            
            if obj.num_cases > 1
                for i = 2:obj.num_cases
                    for j = 1:obj.num_dims
                        assert(isequal(RANS_data(:, j, 1), RANS_data(:, j, i)))
                        assert(isequal(LES_data(:, j, 1), LES_data(:, j, i)))
                    end
                end
            end
        end
        
        function [UP2M_interp, UP2M_interp_tmp] = interp_UP2M(obj, ...
                RANS_x, RANS_y, LES_x, LES_y, UPrime2Mean) % ✓
            % Interpolate LES UPrime2Mean onto RANS grid
            
            UP2M_interp = [];
            for i = 1:obj.num_cases
                UP2M_interp_tmp = [];
                for j = 1:6
                    interp = griddata(LES_x, LES_y, ...
                        UPrime2Mean(:, j, i), RANS_x, RANS_y, 'cubic');
                    UP2M_interp_tmp = cat(2, UP2M_interp_tmp, interp);
                end
                UP2M_interp = cat(3, UP2M_interp, UP2M_interp_tmp);
            end
            
            assert(size(UP2M_interp, 1) == obj.num_RANS_coords)
            assert(size(UP2M_interp, 2) == obj.num_LES_cols - 2)
            assert(size(UP2M_interp, 3) == obj.num_cases)
        end
        
        function nan_list = find_nans_post_interp(obj, UP2M_interp) % ✓
            % Find NaN element indexes
            
            nan_array = [];
            for j = 1:obj.num_cases
                nan_array_tmp = [];
                for i = 1:obj.num_RANS_coords
                    if isnan(UP2M_interp(i, 1, j))
                        nan_array_tmp = cat(1, nan_array_tmp, i);
                    end
                end
                nan_array = cat(2, nan_array, nan_array_tmp);
            end
                
            for i = 1:height(nan_array)
                assert(max(nan_array(i, :)) - min(nan_array(i, :)) == 0)
            end
            
            nan_list = unique(nan_array);
        end
        
        function [RANS_data, UP2M_interp, post_nan_height] = ...
                rm_nans_post_interp(obj, nan_list, RANS_data, UP2M_interp) % ✓
            % Remove NaN values from RANS_data and UP2M_interp
            
            RANS_data(nan_list, :, :) = [];
            UP2M_interp(nan_list, :, :) = [];
            
            post_nan_height = obj.num_RANS_coords - height(nan_list);
            assert(size(RANS_data, 1) == post_nan_height)
            assert(size(UP2M_interp, 1) == size(RANS_data, 1))
            assert(size(RANS_data, 2) == obj.num_RANS_cols)
            assert(size(UP2M_interp, 2) == obj.num_LES_cols - 2)
            assert(size(RANS_data, 3) == obj.num_cases)
            assert(size(UP2M_interp, 3) == obj.num_cases)
            
            assert(sum(isnan(RANS_data), 'all') == 0)
            assert(sum(isnan(UP2M_interp), 'all') == 0)
        end
        
        function flat_array = flatten_array(obj, three_dim_array, ...
                post_nan_height) % ✓
            % Flatten data array to 2D by stacking depth layers
            
            flat_array = [];
            for i = 1:obj.num_cases
                flat_array = cat(1, flat_array, three_dim_array(:, :, i));
            end
            
            assert(size(flat_array, 1) == post_nan_height*obj.num_cases)
        end
        
        function full_dataset = assemble_full(obj, RANS_data_flat, ...
                UP2M_interp_flat, post_nan_height) % ✓
            % Assemble full flat dataset
            
            full_dataset = cat(2, RANS_data_flat, UP2M_interp_flat);
            assert(size(full_dataset, 1) == post_nan_height*obj.num_cases)
            assert(size(full_dataset, 2) == obj.num_RANS_cols + ...
                obj.num_LES_cols - 2 + 3)
            
            randi_1 = randi(post_nan_height*obj.num_cases);
            randi_2 = randi(obj.num_RANS_cols);
            randi_3 = randi(obj.num_LES_cols - 2 + 3);
            assert(RANS_data_flat(randi_1, randi_2) == ...
                full_dataset(randi_1, randi_2))
            assert(UP2M_interp_flat(randi_1, randi_3) == ...
                full_dataset(randi_1, obj.num_RANS_cols + randi_3))
        end
        
        function full_dataset = rm_at_walls(obj, full_dataset, ...
                post_nan_height) % ✓
            % Remove rows corresponding to at-wall results
            
            wall_idx = find(full_dataset(:, 3) == 0);
            assert(length(wall_idx) == obj.num_at_walls*obj.num_cases)
            full_dataset(wall_idx, :) = [];
            
            assert(size(full_dataset, 1) == ...
                (post_nan_height*obj.num_cases) - length(wall_idx))
            assert(size(full_dataset, 2) == obj.num_RANS_cols + ...
                obj.num_LES_cols - 2 + 3)
        end
    end
    
    methods (Static)
        function cmp_interp_plot(RANS_x, RANS_y, LES_x, LES_y, ...
                UP2M_interp_vector, UP2M_LES_vector) % ✓
            % Produce plot of interpolation result and compare with LES
            
            assert(isvector(UP2M_interp_vector))
            assert(isvector(UP2M_LES_vector))
            
            figure();
            scatter(RANS_x, RANS_y, [], UP2M_interp_vector, 'filled')
            figure();
            scatter(LES_x, LES_y, [], UP2M_LES_vector, 'filled')
        end
    end
end

