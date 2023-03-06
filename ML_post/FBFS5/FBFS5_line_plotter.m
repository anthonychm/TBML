%% PREAMBLE %% ✓

% This script produces line plots for component or scalar data

clear variables
top_dir = ['C:\Users\Antho\Dropbox (The University of Manchester)'...
    '\PhD_Anthony_Man\'];
addpath(strcat(top_dir, 'Code\GoTo'));

%% Specify variables ✓

% ***** VARIABLES ***** %

% Data source
inlet_vel = 2;
source = 'tbnn'; % choose: 'tbnn', 'tkenn' or 'tauij'
plot_var = 'b12';
x_loc = 40/3;
child_dir = 'High opt';
zonal = true;

% Constants
x_offset = 0.063;
y_offset = 0.003;
step_height = 0.018;
num_rows = 80296;
num_dims = 2;

% ***** VARIABLES ***** %

%% Extract data for plotting ✓

% Extract reference data indexes
dict_obj = dict_class;
[U_dict, ~] = dict_obj.five_fbfs_case_dict(); %
segment_idx = U_dict(num2str(inlet_vel));
ref_dict = dict_obj.ref_header_dict(); %
var_col_idx = ref_dict(plot_var);

% Extract Cx and Cy
RANS_ref_table = readmatrix('RANS_ref_table.txt');
LES_ref_table = readmatrix('LES_ref_table.txt');
lower = (segment_idx*num_rows) - num_rows + 1;
upper = segment_idx*num_rows;
Cx = RANS_ref_table(lower:upper, 1);
Cy = RANS_ref_table(lower:upper, 2);

% Extract plot variable reference data
RANS_ref_data = RANS_ref_table(lower:upper, var_col_idx);
LES_ref_data = LES_ref_table(lower:upper, var_col_idx);

% Make prediction file column idx dictionary
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

% Read prediction files and extract predicted variable data for plotting
pred = readmatrix(strcat(child_dir, '\U', num2str(inlet_vel), ...
    '_', source, '_one_zone_pred.txt'));
pred_data = pred(:, col_idx_dict(plot_var));

if zonal == true
    zpred = readmatrix(strcat(child_dir, '\U', num2str(inlet_vel), ...
        '_', source, '_zonal_pred_reconstructed.txt'));
    zpred_data = zpred(:, col_idx_dict(plot_var));
end

%% Preprocess Cx and Cy, and find unique x positions ✓
reformat_obj = reformat_class;
[Cx_nd, Cy_nd] = reformat_obj.nondim_fbfs_Cx_Cy(Cx, Cy, x_offset, ...
    y_offset, step_height, num_rows); %
unique_x = unique(Cx_nd);
assert(size(unique_x, 1) == 870)

% Find closest value in Cx_nd to query x_loc
plot_obj = FBFS5_plotter_class;
closest_val = plot_obj.find_closest_val(x_loc, unique_x);

% Extract plotting data
Cy_plot_data = plot_obj.get_spec_Cx_results(closest_val, Cx_nd, Cy_nd);
RANS_plot_data = plot_obj.get_spec_Cx_results(closest_val, Cx_nd, RANS_ref_data);
LES_plot_data = plot_obj.get_spec_Cx_results(closest_val, Cx_nd, LES_ref_data);
pred_plot_data = plot_obj.get_spec_Cx_results(closest_val, Cx_nd, pred_data);
zpred_plot_data = plot_obj.get_spec_Cx_results(closest_val, Cx_nd, zpred_data);

%% Create line plots

figure
plot(Cy_plot_data, RANS_plot_data, 'r', Cy_plot_data, LES_plot_data, 'k',...
    Cy_plot_data, pred_plot_data, 'm', Cy_plot_data, zpred_plot_data, 'b')
legend('RANS', 'LES', 'Non-zonal', 'Zonal')


