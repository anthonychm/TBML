%% PREAMBLE %% ✓✓

% This script produces a table containing tauij, k and bij for both RANS
% and LES results for all FBFS5 cases

clear variables
top_dir = ['C:\Users\Antho\Dropbox (The University of Manchester)\'...
    'PhD_Anthony_Man\'];
data_dir = strcat(top_dir, 'Code\TBNN_workflow\CFD_post\FBFS5\Results\');
addpath(strcat(top_dir, 'Code\GoTo'));
FBFS5_full = readmatrix(strcat(data_dir, 'FBFS5_full_dataset.txt'));

num_rows = 80296;
num_cases = 5;

%% Extract Cx, Cy, RANS tauij, k and bij ✓✓

% Extract idx for NaN and at-wall locations
RANS_cplane_ex = table2array(load(strcat(data_dir, ...
    'solidBlockU1_RANS_cplane_table')).table);
RANS_coords_new = FBFS5_full(1:num_rows, 1:2);
[rm_rows, rm_idx] = setdiff(RANS_cplane_ex(:, 1:2), RANS_coords_new, ...
    'rows');
assert(length(rm_idx) == 1840)

RANS_cplane_ex(rm_idx, :) = [];
assert(isequal(RANS_cplane_ex(:, 1:4), FBFS5_full(1:num_rows, 1:4)));

% Extract RANS tauij data
n = 0;
for i = [1, 2, 2.5, 3, 4]
    n = n + 1;
    RANS_cplane_data = load(strcat(data_dir, 'solidBlockU', ...
        num2str(i), '_RANS_cplane_table.mat'));
    RANS_cplane_data = table2array(RANS_cplane_data.table);
    RANS_cplane_data(rm_idx, :) = [];
    assert(isequal(RANS_cplane_data(:, 1:4), ...
        FBFS5_full(((n*num_rows) - num_rows + 1):n*num_rows, 1:4)));
    if i == 1
        RANS_tauij = RANS_cplane_data(:, end-5:end);
        Cx_Cy = RANS_cplane_data(:, 1:2);
    else
        RANS_tauij = cat(1, RANS_tauij, RANS_cplane_data(:, end-5:end));
        Cx_Cy = cat(1, Cx_Cy, RANS_cplane_data(:, 1:2));
    end
end

% Expand RANS tauij to 9 components
reformat_obj = reformat_class;
RANS_tauij = reformat_obj.symmTensor2tensor(RANS_tauij);
assert(size(RANS_tauij, 1) == num_cases*num_rows);
assert(size(RANS_tauij, 2) == 9);

% Extract RANS k
RANS_k = FBFS5_full(:, 3);
assert(size(RANS_k, 1) == num_cases*num_rows)

% Calculate RANS bij
calc_obj = calc_class;
RANS_bij = calc_obj.calc_bij_from_tauij(RANS_tauij);
assert(size(RANS_bij, 1) == num_cases*num_rows)

%% Extract LES tauij, k and bij ✓✓

% Extract LES_tauij
LES_tauij = FBFS5_full(:, end-8:end);
assert(size(LES_tauij, 1) == num_cases*num_rows)

% Calculate LES k
LES_k = calc_obj.calc_k_from_tauij(LES_tauij);
assert(size(LES_k, 1) == num_cases*num_rows)

% Calculate LES bij
LES_bij = calc_obj.calc_bij_from_tauij(LES_tauij);
assert(size(LES_bij, 1) == num_cases*num_rows)

%% Build and write tables ✓✓

% Build tables (col order: Cx, Cy, bij, k, tauij)
RANS_table = cat(2, Cx_Cy, RANS_bij, RANS_k, RANS_tauij);
LES_table = cat(2, Cx_Cy, LES_bij, LES_k, LES_tauij);

% Write tables to file
header = {'Cx', 'Cy', 'b11', 'b12', 'b13', 'b21', 'b22', 'b23', 'b31', ...
    'b32', 'b33', 'k', 'tau11', 'tau12', 'tau13', 'tau21', 'tau22', ...
    'tau23', 'tau31', 'tau32', 'tau33'};
RANS_table = array2table(RANS_table, 'VariableNames', header);
writetable(RANS_table, strcat('RANS_ref_table.txt'), 'Delimiter', ' ')
LES_table = array2table(LES_table, 'VariableNames', header);
writetable(LES_table, strcat('LES_ref_table.txt'), 'Delimiter', ' ')
