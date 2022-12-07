%% PREAMBLE %%
% This script is the successor of FBFS_TBNN_concatenator.m and
% ChannelFlow_TBNN_concatenator.m

% This script concatenates RANS input data and LES ground truth output
% results into one array for the TBNN Python code to read.

%% Define parameters, objects and attributes ✓✓

clear variables
top_dir = ['C:\Users\Antho\Dropbox (The University of Manchester)\'...
    'PhD_Anthony_Man\'];
addpath(strcat(top_dir, 'Code\GoTo'), '..\')
concat_obj = CFD_data_concat_class;
reformat_obj = reformat_class;

concat_obj.param_values = [1, 2, 2.5, 3, 4];
concat_obj.case_prefix = 'solidBlockU';
concat_obj.in_table_name = 'cplane_table';
concat_obj.num_RANS_coords = 82136;
concat_obj.num_LES_coords = 106800;
concat_obj.num_RANS_cols = 23;
concat_obj.num_LES_cols = 8;
concat_obj.num_cases = length(concat_obj.param_values);
concat_obj.num_dims = 2;
concat_obj.num_at_walls = 1832;

%% Read RANS and LES data ✓✓

% Initialise data arrays ✓
RANS_data = NaN(concat_obj.num_RANS_coords, concat_obj.num_RANS_cols, ...
    concat_obj.num_cases);
LES_data = NaN(concat_obj.num_LES_coords, concat_obj.num_LES_cols, ...
    concat_obj.num_cases);

% Fill data arrays ✓
[RANS_data, LES_data] = read_CFD_data(concat_obj, RANS_data, LES_data); % ✓
init_assert_checks(concat_obj, RANS_data, LES_data) % ✓
disp('CFD data reading completed')

%% Interpolate LES UPrime2Mean onto RANS grid ✓✓

% Specify RANS and LES coordinate points ✓
% (init_assert_checks shows x and y are the same across all depth layers, 
%  so we can take x and y from the first layer in RANS_data and LES_data)
RANS_x = RANS_data(:, 1, 1);
RANS_y = RANS_data(:, 2, 1);
LES_x = LES_data(:, 1, 1);
LES_y = LES_data(:, 2, 1);
UPrime2Mean = LES_data(:, 3:end, :);

% Execute interpolation ✓
[UP2M_interp, UP2M_interp_tmp] = interp_UP2M(concat_obj, RANS_x, ...
    RANS_y, LES_x, LES_y, UPrime2Mean); % ✓
disp('LES UPrime2Mean interpolated onto RANS grid')

% Compare interpolated results with LES UPrime2Mean ✓
concat_obj.cmp_interp_plot(RANS_x, RANS_y, LES_x, LES_y, ...
    UP2M_interp(:, 1, 1), UPrime2Mean(:, 1, 1)) % ✓ U1 u'u'
concat_obj.cmp_interp_plot(RANS_x, RANS_y, LES_x, LES_y, ...
    UP2M_interp(:, 4, 2), UPrime2Mean(:, 4, 2)) % ✓ U2 v'v'
concat_obj.cmp_interp_plot(RANS_x, RANS_y, LES_x, LES_y, ...
    UP2M_interp(:, 2, 4), UPrime2Mean(:, 2, 4)) % ✓ U3 u'v'

%% Remove NaN values from RANS_data and UP2M_interp ✓✓

% Find NaN element indexes ✓
nan_list = find_nans_post_interp(concat_obj, UP2M_interp); % ✓

% Remove NaN values ✓
[RANS_data, UP2M_interp, post_nan_height] = rm_nans_post_interp(...
    concat_obj, nan_list, RANS_data, UP2M_interp); % ✓

%% Flatten arrays from 3D to vertical stacked 2D ✓✓

RANS_data_flat = flatten_array(concat_obj, RANS_data, post_nan_height); % ✓
UP2M_interp_flat = flatten_array(concat_obj, UP2M_interp, post_nan_height); % ✓

assert(size(RANS_data_flat, 2) == concat_obj.num_RANS_cols)
assert(size(UP2M_interp_flat, 2) == concat_obj.num_LES_cols - 2)

%% Assemble full flat dataset ✓✓

% Include symmetric UPrime2Mean components in UP2M_interp_flat ✓
UP2M_interp_flat = reformat_class.symmTensor2tensor(UP2M_interp_flat); % ✓

% Assemble full flat dataset ✓
full_dataset = assemble_full(concat_obj, RANS_data_flat, ...
    UP2M_interp_flat, post_nan_height); % ✓

% Remove at-wall rows ✓
full_dataset = rm_at_walls(concat_obj, full_dataset, post_nan_height); % ✓
disp('Full dataset assembled')

%% Write full flat dataset to file ✓✓

% This code section is correct for PoF Oct 2022. For future work, please
% check the following header for compatibility.

header = {'Cx', 'Cy', 'tke', 'epsilon', 'dU/dx', 'dU/dy', 'dU/dz', ...
    'dV/dx', 'dV/dy', 'dV/dz', 'dW/dx', 'dW/dy', 'dW/dz', 'dp/dx', ...
    'dp/dy', 'dp/dz', 'Ux', 'Uy', 'Uz', 'dk/dx', 'dk/dy', 'dk/dz', ...
    'nondim_tke', 'uu_11', 'uu_12', 'uu_13', 'uu_21', 'uu_22', 'uu_23', ...
    'uu_31', 'uu_32', 'uu_33'};

full_dataset = array2table(full_dataset, 'VariableNames', header);
writetable(full_dataset, 'FBFS5_full_dataset.txt', 'Delimiter', ' ')
disp('full flat dataset written to directory')
