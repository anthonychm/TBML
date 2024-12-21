%% PREAMBLE %% ✓✓

% This script plots the Barycentric Map for bij data, taken from a line in
% mesh array format
clear variables
addpath('..\')
bary_obj = bary_map_class;
plot_obj = FBFS5_plotter_class;

%% Specify variables ✓✓

% ***** VARIABLES ***** %
inlet_vel = 4;

% choose: RANS, LES, non-zonal, smooth_non-zonal, zonal, smooth_zonal
source = 'non-zonal';
z_source = 'smooth_zonal';
sample_dir = 'x'; % sample direction, choose: x or y
sample_const_pos = 1;
sample_bounds = [7.5, 10];

marker = 'c^';
z_marker = 'go';
RANS_marker = 'rsquare';
LES_marker = 'ydiamond';
%marker_size = 5;
% ***** VARIABLES ***** %

%% Find line extraction idx ✓✓

% Load nondimensional RANS mesh_x and mesh_y and calculate unique_x and 
% unique_y ✓
mesh_x = struct2cell(load('RANS mesh format results\RANS_nondim_mesh_x.mat'));
mesh_x = mesh_x{1};
unique_x = unique(mesh_x);

mesh_y = struct2cell(load('RANS mesh format results\RANS_nondim_mesh_y.mat'));
mesh_y = mesh_y{1};
unique_y = unique(mesh_y);

% Extract start and end x idx, and constant y idx ✓
if strcmp(sample_dir, 'x')
    closest_start_x = plot_obj.find_closest_val(sample_bounds(1), unique_x); % ✓
    [~, start_idx] = find(mesh_x == closest_start_x, 1, 'first');
    closest_end_x = plot_obj.find_closest_val(sample_bounds(2), unique_x); % ✓
    [~, end_idx] = find(mesh_x == closest_end_x, 1, 'first');
    closest_const_pos = plot_obj.find_closest_val(sample_const_pos, unique_y); % ✓
    [const_idx, ~] = find(mesh_y == closest_const_pos, 1, 'first');
% Extract start and end y idx, and constant x idx ✓
elseif strcmp(sample_dir, 'y')
    closest_start_y = plot_obj.find_closest_val(sample_bounds(1), unique_y); % ✓
    [start_idx, ~] = find(mesh_y == closest_start_y, 1, 'first');
    closest_end_y = plot_obj.find_closest_val(sample_bounds(2), unique_y); % ✓
    [end_idx, ~] = find(mesh_y == closest_end_y, 1, 'first');
    closest_const_pos = plot_obj.find_closest_val(sample_const_pos, unique_x); % ✓
    [~, const_idx] = find(mesh_x == closest_const_pos, 1, 'first');
else
    error('invalid sample_dir')
end

%% Load mesh line data and reformat into 3 x 3 x num points bij arrays ✓✓

% Load non-zonal prediction data ✓
data_folder = strcat('High opt\Mesh format results\U', ...
    num2str(inlet_vel), '_', source, '\');
[b11, b12, b13, b21, b22, b23, b31, b32, b33] = bary_obj.load_mesh_line(...
    inlet_vel, source, data_folder, sample_dir, start_idx, end_idx, const_idx); % ✓

% Load zonal prediction data ✓
z_data_folder = strcat('High opt\Mesh format results\U', ...
    num2str(inlet_vel), '_', z_source, '\');
[zb11, zb12, zb13, zb21, zb22, zb23, zb31, zb32, zb33] = bary_obj.load_mesh_line(...
    inlet_vel, z_source, z_data_folder, sample_dir, start_idx, end_idx, const_idx); % ✓

% Load RANS data ✓
RANS_folder = strcat('RANS mesh format results\U', num2str(inlet_vel), '\');
[RANS_b11, RANS_b12, RANS_b13, RANS_b21, RANS_b22, RANS_b23, RANS_b31, ...
    RANS_b32, RANS_b33] = bary_obj.load_mesh_line(inlet_vel, 'RANS', ...
    RANS_folder, sample_dir, start_idx, end_idx, const_idx); % ✓

% Load LES data ✓
LES_folder = strcat('LES mesh format results\U', num2str(inlet_vel), '\');
[LES_b11, LES_b12, LES_b13, LES_b21, LES_b22, LES_b23, LES_b31, ...
    LES_b32, LES_b33] = bary_obj.load_mesh_line(inlet_vel, 'LES', ...
    LES_folder, sample_dir, start_idx, end_idx, const_idx); % ✓

% Reformat bij data into 3 x 3 x num points array ✓
pred_bij = bary_obj.reformat_three_by_three(b11, b12, b13, b21, b22, ...
    b23, b31, b32, b33); % ✓
zpred_bij = bary_obj.reformat_three_by_three(zb11, zb12, zb13, zb21, zb22, ...
    zb23, zb31, zb32, zb33); % ✓
RANS_bij = bary_obj.reformat_three_by_three(RANS_b11, RANS_b12, ...
    RANS_b13, RANS_b21, RANS_b22, RANS_b23, RANS_b31, RANS_b32, RANS_b33); % ✓
LES_bij = bary_obj.reformat_three_by_three(LES_b11, LES_b12, LES_b13, ...
    LES_b21, LES_b22, LES_b23, LES_b31, LES_b32, LES_b33); % ✓
            
%% Establish limiting states ✓✓

% One-component state
lambdaA = [(2/3), 0, 0; 0, (-1/3), 0; 0, 0, (-1/3)];
[CA1, CA2, CA3] = bary_obj.calc_coeffs(lambdaA); % ✓

% Two-component state
lambdaB = [(1/6), 0, 0; 0, (1/6), 0; 0, 0, (-1/3)];
[CB1, CB2, CB3] = bary_obj.calc_coeffs(lambdaB); % ✓

% Three-component state
lambdaC = zeros(3, 3);
[CC1, CC2, CC3] = bary_obj.calc_coeffs(lambdaC); % ✓

% One-component state coordinates
A_x = 1;
A_y = 0;

% Two-component state coordinates
B_x = 0;
B_y = 0;

% Three-component state coordinates
C_x = 1/2;
C_y = sqrt(3)/2;

% Calculate border line equations
% y = 0
% y = (C_y - B_y)/(C_x - B_x)x
% y = (A_y - C_y)/(A_x - C_x)x + -(A_y - C_y)/(A_x - C_x)


%% Create Barycentric map ✓✓

% Set Barycentric map background
bary_obj.create_tri(A_x, A_y, B_x, B_y, C_x, C_y) % ✓

% Find eigenvalues, coeffs and plot (x_bary, y_bary) for each bij slice ✓
pred_flag_list = bary_obj.plot_bary_coords(pred_bij, ...
    A_x, A_y, B_x, B_y, C_x, C_y, marker); % ✓
zpred_flag_list = bary_obj.plot_bary_coords(zpred_bij, ...
    A_x, A_y, B_x, B_y, C_x, C_y, z_marker); % ✓
RANS_flag_list = bary_obj.plot_bary_coords(RANS_bij, ...
    A_x, A_y, B_x, B_y, C_x, C_y, RANS_marker); % ✓
LES_flag_list = bary_obj.plot_bary_coords(LES_bij, ...
    A_x, A_y, B_x, B_y, C_x, C_y, LES_marker); % ✓

% xlim([-0.5 1.5])
% ylim([-0.5 1.5])

xlim([0 1])
ylim([0 1])
% xlim([-0.15 1])
% ylim([-0.15 1])
% xticks(0:0.2:1)
% yticks(0:0.2:1)

%% Quantitative analysis

% Find mean BM coordinates
nz_mean_coords = bary_obj.calc_mean_coords(nz_coords);
z_mean_coords = bary_obj.calc_mean_coords(z_coords);
RANS_mean_coords = bary_obj.calc_mean_coords(RANS_coords);
LES_mean_coords = bary_obj.calc_mean_coords(LES_coords);

% Find mean unit vector
nz_mean_unit_vec = bary_obj.calc_mean_unit_vec(nz_coords);
z_mean_unit_vec = bary_obj.calc_mean_unit_vec(z_coords);
RANS_mean_unit_vec = bary_obj.calc_mean_unit_vec(RANS_coords);
LES_mean_unit_vec = bary_obj.calc_mean_unit_vec(LES_coords);

%% Include mean coordinates in BM

% Include mean coordinates
plot(nz_mean_coords(1), nz_mean_coords(2), 'k^', ...
    'MarkerSize', 2+(size(nz_coords, 1)*0.5*0.15), 'LineWidth', 2)
plot(z_mean_coords(1), z_mean_coords(2), 'ko', ...
    'MarkerSize', 2+(size(z_coords, 1)*0.5*0.15), 'LineWidth', 2)
plot(RANS_mean_coords(1), RANS_mean_coords(2), 'ksquare', ...
    'MarkerSize', 2+(size(RANS_coords, 1)*0.5*0.15), 'LineWidth', 2)
plot(LES_mean_coords(1), LES_mean_coords(2), 'kdiamond', ...
    'MarkerSize', 2+(size(LES_coords, 1)*0.5*0.15), 'LineWidth', 2)

%% Test: Plot the three components states ✓

% % One-component state
% [x_bary, y_bary] = bary_obj.calc_coords(CA1, A_x, A_y, CA2, B_x, B_y, ...
%     CA3, C_x, C_y); % ✓
% plot(x_bary, y_bary, '.', 'MarkerSize', marker_size)
% 
% % Two-component state
% [x_bary, y_bary] = bary_obj.calc_coords(CB1, A_x, A_y, CB2, B_x, B_y, ...
%     CB3, C_x, C_y); % ✓
% plot(x_bary, y_bary, '.', 'MarkerSize', marker_size)
% 
% % Three-component state
% [x_bary, y_bary] = bary_obj.calc_coords(CC1, A_x, A_y, CC2, B_x, B_y, ...
%     CC3, C_x, C_y); % ✓
% plot(x_bary, y_bary, '.', 'MarkerSize', marker_size)




