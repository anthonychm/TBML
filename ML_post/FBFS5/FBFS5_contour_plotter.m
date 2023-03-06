%% PREAMBLE %% ✓✓

% This script produces contour plots for bij/k/tauij zonal/non-zonal data

clear variables
top_dir = ['C:\Users\Antho\Dropbox (The University of Manchester)'...
    '\PhD_Anthony_Man\'];
addpath(strcat(top_dir, 'Code\GoTo'));

%% Specify variables ✓✓

% ***** VARIABLES ***** %

% Data source
inlet_vel = 4;
source = 'tbnn'; % choose: 'tbnn', 'tkenn' or 'tauij'
plot_var = 'b33';
child_dir = 'High opt';
zonal = true;

% Constants
x_offset = 0.063;
y_offset = 0.003;
step_height = 0.018;
num_rows = 80296;
num_dims = 2;

% Contour comparison plot settings
num_colours = 200;

% Error plot settings
error_var = 'bij'; % choose: 'bij', 'k' or 'tauij'
query_sim = 'zonal'; % choose: 'RANS', 'one_zone' or 'zonal'
base_sim = 'LES'; % choose: 'RANS', 'LES'

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
if strcmp(source, 'tbnn') 
    keys = {'b11', 'b12', 'b13', 'b21', 'b22', 'b23', 'b31', 'b32', 'b33'};
elseif strcmp(source, 'tauij')
    keys = {'tau11', 'tau12', 'tau13', 'tau21', 'tau22', 'tau23', ...
        'tau31', 'tau32', 'tau33'};
elseif strcmp(source, 'tkenn')
    keys = {'log(k)', 'k'};
else
    error('invalid source');
end

col_idx_dict = containers.Map(keys, ...
    (num_dims + 1):(num_dims + length(keys)));

% Read prediction files and extract predicted variable data for plotting ✓
pred = readmatrix(strcat(child_dir, '\U', num2str(inlet_vel), ...
    '_', source, '_one_zone_pred.txt'));
pred_data = pred(:, col_idx_dict(plot_var));

if zonal == true
    zpred = readmatrix(strcat(child_dir, '\U', num2str(inlet_vel), ...
        '_', source, '_zonal_pred_reconstructed.txt'));
    zpred_data = zpred(:, col_idx_dict(plot_var));
end

%% Single component or scalar vs RANS, LES and One-zone contours ✓✓

% Prepare contour mesh ✓
reformat_obj = reformat_class;
[Cx_nd, Cy_nd] = reformat_obj.nondim_fbfs_Cx_Cy(Cx, Cy, x_offset, ...
    y_offset, step_height, num_rows); % ✓
contour_obj = contour_plot_class;
contour_obj.Cx = Cx_nd;
contour_obj.Cy = Cy_nd;
[unique_x, unique_y, mesh_x, mesh_y] = create_contour_mesh(contour_obj); % ✓

% Prepare component or scalar plot data ✓
RANS_plot_data = extract_var_contour_data(contour_obj, unique_x, ...
    unique_y, mesh_x, RANS_ref_data); % ✓
LES_plot_data = extract_var_contour_data(contour_obj, unique_x, ...
    unique_y, mesh_x, LES_ref_data); % ✓
one_zone_plot_data = extract_var_contour_data(contour_obj, unique_x, ...
    unique_y, mesh_x, pred_data); % ✓
plot_data = cat(1, RANS_plot_data, LES_plot_data, one_zone_plot_data);
if zonal == true
    zonal_plot_data = extract_var_contour_data(contour_obj, unique_x, ...
        unique_y, mesh_x, zpred_data); % ✓
    plot_data = cat(1, plot_data, zonal_plot_data);
end

% Produce contour plots (3 or 4 x 1) ✓
plot_obj = FBFS5_plotter_class;
plot_obj.contour(mesh_x, mesh_y, plot_data, num_colours, plot_var, zonal); % ✓

%% Error contour vs LES

% Extract error query data

% If RANS:
if strcmp(query_sim, 'RANS')
    query_data = plot_obj.extract_from_ref(query_sim, error_var, ...
        lower, upper);

% If one-zone or zonal:
elseif strcmp(query_sim, 'one_zone') || strcmp(query_sim, 'zonal')
    file_dict = containers.Map({'bij', 'k', 'tauij'}, ...
        {'tbnn', 'tkenn', 'tauij'});
    query_data = strcat(child_dir, '\U', num2str(inlet_vel), '_', ...
        file_dict(error_var), '_', query_sim, '_pred');
    if strcmp(query_sim, 'zonal')
        query_data = strcat(query_data, '_reconstructed');
    end
    query_data = readmatrix(query_data);
    
    % If k, else bij or tauij:
    if strcmp(error_var, 'k')
        query_data = query_data(:, end);
    else
        query_data = query_data(:, end-8:end);
    end
else
    error('invalid simulation query');
end

% Extract error base data
base_data = plot_obj.extract_from_ref(base_sim, error_var, lower, upper);

% Prepare error plot data
calc_obj = calc_class;
[error, ~] = calc_obj.error_calc(query_data, base_data);
error_plot_data = extract_error_contour_data(contour_obj, unique_x, ...
    unique_y, mesh_x, error);

% Produce error contour plots
error_contour(mesh_x, mesh_y, error_plot_data, num_colours);


