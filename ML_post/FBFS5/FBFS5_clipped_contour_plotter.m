%% PREAMBLE %% ✓✓

% This script produces clipped contour plots for non-zonal/zonal bij/k data

clear variables
top_dir = ['C:\Users\Antho\Dropbox (The University of Manchester)'...
    '\PhD_Anthony_Man\'];
addpath(strcat(top_dir, 'Code\GoTo'));

%% Specify variables ✓✓

% ***** VARIABLES ***** %

% Data source settings for all plots and error calculation
child_dir = 'High opt';
inlet_vel = 2;
nn = 'tkenn'; % choose: 'tbnn' or 'tkenn'
plot_var = 'k';
x_start = -1; % -1 or 10/3
x_end = 71/3; % 71/3 or 12.1

% First contour tile data source settings
% choose: 'one_zone_pred' or 'zonal_pred_reconstructed'
first_tile = 'zonal_pred_reconstructed';
first_smooth = false;

% Second contour tile data source settings
plot_second_tile = false;
% choose: 'one_zone_pred' or 'zonal_pred_reconstructed'
second_tile = 'zonal_pred_reconstructed';
second_smooth = true;

% Contour plot settings
num_colours = 200;

% Line plot settings
line_x_pos = 10; % 4, 5, 8, 10
plot_second_line = true;

% Error settings
query_sim = 'first_pred'; % choose: 'RANS' or 'first_pred'
base_sim = 'LES'; % choose: 'RANS', 'LES'
error_region = 'whole'; % choose: 'zone1', 'zone2' or 'whole'

% Constants
x_offset = 0.063;
y_offset = 0.003;
step_height = 0.018;
num_rows = 80296;
num_dims = 2;

% ***** VARIABLES ***** %

%% Extract data for plotting ✓✓

% Extract reference data indexes ✓
dict_obj = dict_class;
[U_dict, ~] = dict_obj.five_fbfs_case_dict(); % ✓
segment_idx = U_dict(num2str(inlet_vel));
ref_dict = dict_obj.ref_header_dict(); % ✓
var_col_idx = ref_dict(plot_var);

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

% Make prediction file column idx dictionary ✓
if strcmp(nn, 'tbnn') 
    keys = {'b11', 'b12', 'b13', 'b21', 'b22', 'b23', 'b31', 'b32', 'b33'};
elseif strcmp(nn, 'tkenn')
    keys = {'log(k)', 'k'};
else
    error('invalid neural network source');
end

col_idx_dict = containers.Map(keys, ...
    (num_dims + 1):(num_dims + length(keys)));

% Read prediction files and extract predicted variable data for plotting ✓
first_pred = readmatrix(strcat(child_dir, '\U', num2str(inlet_vel), ...
    '_', nn, '_', first_tile, '.txt'));
first_pred_data = first_pred(:, col_idx_dict(plot_var));

if plot_second_tile == true
    second_pred = readmatrix(strcat(child_dir, '\U', ...
        num2str(inlet_vel), '_', nn, '_', second_tile, '.txt'));
    second_pred_data = second_pred(:, col_idx_dict(plot_var));
end

% Prepare contour mesh ✓
reformat_obj = reformat_class;
[Cx_nd, Cy_nd] = reformat_obj.nondim_fbfs_Cx_Cy(Cx, Cy, x_offset, ...
    y_offset, step_height, num_rows); % ✓
contour_obj = contour_plot_class;
contour_obj.Cx = Cx_nd;
contour_obj.Cy = Cy_nd;
[unique_x, unique_y, mesh_x, mesh_y] = create_contour_mesh(contour_obj); % ✓

% Prepare data for plotting ✓
RANS_plot_data = extract_var_contour_data(contour_obj, unique_x, ...
    unique_y, mesh_x, RANS_ref_data); % ✓
LES_plot_data = extract_var_contour_data(contour_obj, unique_x, ...
    unique_y, mesh_x, LES_ref_data); % ✓
first_plot_data = extract_var_contour_data(contour_obj, unique_x, ...
    unique_y, mesh_x, first_pred_data); % ✓
if plot_second_tile == true
    second_plot_data = extract_var_contour_data(contour_obj, unique_x, ...
        unique_y, mesh_x, second_pred_data); % ✓
end

%% Clip mesh data arrays ✓✓

% Find col idx closest to start and end x clip positions ✓
plot_obj = FBFS5_plotter_class;
closest_start_val = plot_obj.find_closest_val(x_start, unique_x); % ✓
closest_end_val = plot_obj.find_closest_val(x_end, unique_x); % ✓
[~, closest_start_col_idx] = find(mesh_x == closest_start_val, 1, 'first');
[~, closest_end_col_idx] = find(mesh_x == closest_end_val, 1, 'first');

% Clip plot_data arrays according to x_start and x_end ✓
mesh_xc = mesh_x(:, closest_start_col_idx:closest_end_col_idx);
mesh_yc = mesh_y(:, closest_start_col_idx:closest_end_col_idx);
RANS_plot_datac = RANS_plot_data(:, ...
    closest_start_col_idx:closest_end_col_idx);
LES_plot_datac = LES_plot_data(:, ...
    closest_start_col_idx:closest_end_col_idx);
first_plot_datac = first_plot_data(:, ...
    closest_start_col_idx:closest_end_col_idx);

% Smooth first_plot_datac and create plot_datac array ✓
if first_smooth == true
    first_plot_datac = plot_obj.smooth_data(first_plot_datac, ...
        'gaussian', 10); % ✓
end
plot_datac = cat(1, RANS_plot_datac, LES_plot_datac, first_plot_datac);

% Smooth second_plot_datac and concatenate to plot_datac array ✓
if plot_second_tile == true
    second_plot_datac = second_plot_data(:, ...
        closest_start_col_idx:closest_end_col_idx);
    if second_smooth == true
        second_plot_datac = plot_obj.smooth_data(second_plot_datac, ...
            'gaussian', 10); % ✓
    end
    plot_datac = cat(1, plot_datac, second_plot_datac);
end

%% Write first_plot_data for Barycentric map and tauij plots ✓

% assert(x_start == -1)
% assert(x_end == 71/3)
% save(strcat('U', num2str(inlet_vel), '_', 'RANS', '_', plot_var, ...
%     '_mesh.mat'), 'RANS_plot_datac');
% save(strcat('U', num2str(inlet_vel), '_', 'LES', '_', plot_var, ...
%     '_mesh.mat'), 'LES_plot_datac');

%% Produce clipped contour plots (3 or 4 x 1) ✓✓

plot_obj.contour(mesh_xc, mesh_yc, plot_datac, num_colours, plot_var, ...
    plot_second_tile); % ✓

%% Produce line plots for specific x position ✓✓

% assert(x_start <= line_x_pos)
% assert(line_x_pos <= x_end)
% 
% % Extract line data ✓
% closest_val = plot_obj.find_closest_val(line_x_pos, unique_x); % ✓
% [~, closest_col_idx] = find(mesh_xc == closest_val, 1, 'first');
% y_line_data = mesh_yc(:, closest_col_idx);
% RANS_line_data = RANS_plot_datac(:, closest_col_idx);
% LES_line_data = LES_plot_datac(:, closest_col_idx);
% first_line_data = first_plot_datac(:, closest_col_idx);
% 
% % Produce line plots ✓
% if plot_second_line == true
%     second_line_data = second_plot_datac(:, closest_col_idx);
%     figure
%     hold on
%     plot(RANS_line_data, y_line_data, 'r--', LES_line_data, y_line_data, ...
%         'k-.', first_line_data, y_line_data, 'b:' , 'LineWidth', 2)
%     plot(second_line_data, y_line_data, 'Color', '#00B000', 'LineWidth', 2)
%     legend('RANS', 'LES', 'Non-zonal', 'Zonal')
%     xlabel('b_1_1')
%     ylabel('y/h')
%     set(gcf, 'Position',  [100, 100, 200, 200])
%     grid on
% else
%     figure
%     plot(y_line_data, RANS_line_data, 'r', y_line_data, LES_line_data, ...
%         'k', y_line_data, first_line_data, 'm')
%     legend('RANS', 'LES', 'First pred')
% end

%% Calculate error statistics ✓✓

% % Extract base and query data ✓
% assert(x_start == -1)
% assert(x_end == 71/3)
% error_obj = FBFS5_error_class;
% if strcmp(query_sim, 'RANS')
%     query_data = RANS_plot_data;
% elseif strcmp(query_sim, 'first_pred')
%     query_data = first_plot_datac;
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

