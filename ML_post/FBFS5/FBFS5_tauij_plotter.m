%% PREAMBLE %% ✓✓

% This script reconstructs Reynolds stress tauij from bij and k predictions
% by TBNN and TKENN respectively. Predicted bij and k must be in mesh 
% format for the contour plotter in this script.

clear variables
top_dir = ['C:\Users\Antho\Dropbox (The University of Manchester)'...
    '\PhD_Anthony_Man\'];
addpath(strcat(top_dir, 'Code\GoTo'));

%% Specify variables ✓✓

% ***** VARIABLES ***** %

% Data source
child_dir = 'High opt\Mesh format results';
inlet_vel = 2; % choose: 2 or 4
comp = 'tau11';
% choose: non-zonal, smooth_non-zonal, zonal, smooth_zonal
source = 'smooth_zonal';
x_start = -1;
x_end = 71/3;

% Constants
x_offset = 0.063;
y_offset = 0.003;
step_height = 0.018;
num_rows = 80296;

% Contour comparison plot settings
num_colours = 200;

% Error settings
query_sim = 'source'; % choose: 'RANS' or 'source'
base_sim = 'LES'; % choose: 'RANS', 'LES'
error_region = 'zone2'; % choose: 'zone1', 'zone2' or 'whole'

% ***** VARIABLES ***** %

%% Set up RANS and LES tauij data in mesh format ✓✓

% Import RANS and LES data ✓
dict_obj = dict_class;
[U_dict, ~] = dict_obj.five_fbfs_case_dict(); % ✓
segment_idx = U_dict(num2str(inlet_vel));
ref_dict = dict_obj.ref_header_dict(); % ✓
var_col_idx = ref_dict(comp);

% Extract Cx and Cy ✓
RANS_ref_table = readmatrix('RANS_ref_table.txt');
LES_ref_table = readmatrix('LES_ref_table.txt');
lower = (segment_idx*num_rows) - num_rows + 1;
upper = segment_idx*num_rows;
Cx = RANS_ref_table(lower:upper, 1);
Cy = RANS_ref_table(lower:upper, 2);

% Extract plot variable reference data ✓
RANS_ref_data = RANS_ref_table(lower:upper, var_col_idx);
LES_ref_data = LES_ref_table(lower:upper, var_col_idx);
reformat_obj = reformat_class;
[Cx_nd, Cy_nd] = reformat_obj.nondim_fbfs_Cx_Cy(Cx, Cy, x_offset, ...
    y_offset, step_height, num_rows); % ✓

% Prepare contour mesh ✓
contour_obj = contour_plot_class;
contour_obj.Cx = Cx_nd;
contour_obj.Cy = Cy_nd;
[unique_x, unique_y, mesh_x, mesh_y] = create_contour_mesh(contour_obj); % ✓

% Prepare data for plotting ✓
RANS_plot_data = extract_var_contour_data(contour_obj, unique_x, ...
    unique_y, mesh_x, RANS_ref_data); % ✓
LES_plot_data = extract_var_contour_data(contour_obj, unique_x, ...
    unique_y, mesh_x, LES_ref_data); % ✓

%% Reconstruct predicted tauij ✓✓

% Import prediction data in mesh format ✓
comp_nums = cell2mat(extract(comp, digitsPattern));
bij_data = struct2cell(load(strcat(child_dir, '\U', num2str(inlet_vel), ...
    '_', source, '\U', num2str(inlet_vel), '_', source, '_b', ...
    comp_nums, '_mesh.mat')));
k_data = struct2cell(load(strcat(child_dir, '\U', num2str(inlet_vel), ...
    '_', source, '\U', num2str(inlet_vel), '_', source, '_k_mesh.mat')));
bij_data = bij_data{1};
k_data = k_data{1};

% Calculate reconstructed tauij component ✓
% Note: i, j iterators here are not the same as ij components
tauij = NaN(size(bij_data, 1), size(bij_data, 2));
for i = 1:size(bij_data, 1)
    for j = 1:size(bij_data, 2)
        if ~isnan(bij_data(i, j))
            if strcmp(comp_nums(1), comp_nums(2))
                tauij(i, j) = 2*k_data(i, j)*(bij_data(i, j) + (1/3));
            else
                tauij(i, j) = 2*k_data(i, j)*(bij_data(i, j));
            end
        end
    end
end

%% Plot tauij contours ✓✓

plot_data = cat(1, RANS_plot_data, LES_plot_data, tauij);
plot_obj = FBFS5_plotter_class;

% Clip plot_data arrays according to x_start and x_end ✓
closest_start_val = plot_obj.find_closest_val(x_start, Cx_nd); % ✓
closest_end_val = plot_obj.find_closest_val(x_end, Cx_nd); % ✓
[~, closest_start_col_idx] = find(mesh_x == closest_start_val, 1, 'first');
[~, closest_end_col_idx] = find(mesh_x == closest_end_val, 1, 'first');

mesh_xc = mesh_x(:, closest_start_col_idx:closest_end_col_idx);
mesh_yc = mesh_y(:, closest_start_col_idx:closest_end_col_idx);
plot_datac = plot_data(:, closest_start_col_idx:closest_end_col_idx);

% Produce contour plots ✓
plot_obj.contour(mesh_xc, mesh_yc, plot_datac, num_colours, comp, false); % ✓

%% Calculate tauij errors ✓✓

% % Extract base and query data ✓
% error_obj = FBFS5_error_class;
% if strcmp(query_sim, 'RANS')
%     query_data = RANS_plot_data;
% elseif strcmp(query_sim, 'source')
%     query_data = tauij;
% else
%     error('invalid query data')
% end
% base_data = error_obj.extract_base_data(base_sim, RANS_plot_data, ...
%     LES_plot_data); % ✓
% 
% % Calculate error, MSE and RMSE ✓
% calc_obj = calc_class;
% [error, ~] = calc_obj.error_calc(query_data, base_data); % ✓
% 
% if strcmp(error_region, 'zone1') || strcmp(error_region, 'zone2')
%     [Cx, Cy] = error_obj.extract_Cx_Cy(error_region, inlet_vel, top_dir); % ✓
%     num_rows = size(Cx, 1);
%     [Cx_nd, Cy_nd] = reformat_obj.nondim_fbfs_Cx_Cy(Cx, Cy, x_offset, ...
%         y_offset, step_height, num_rows); % ✓
% end
% 
% [error_sq, mse, rmse] = calc_obj.mesh_mse_calc(error, Cx_nd, Cy_nd, ...
%     mesh_x, mesh_y); % ✓