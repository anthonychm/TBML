%% PREAMBLE %% ✓✓

% This script reconstructs zonal prediction results according to their 
% respective cases

clear variables
top_dir = ['C:\Users\Antho\Dropbox (The University of Manchester)'...
    '\PhD_Anthony_Man\Code\TBNN_workflow\Driver\Zonal high opt pfalse '...
    'kfalse\'];
addpath('..\')
num_rows = 80296;

%% Execute reconstruction ✓✓

% Prepare zonal reconstruction object and read zonal prediction data ✓
zrecon_obj = zonal_reconstruct_class;
zrecon_obj.num_splits = 1;
z1_tbnn_pred = readmatrix(strcat(top_dir, ['Zone 1 TBNN output data\'...
    'Trials\Trial 1\Trial1_seed1_TBNN_test_prediction_bij.txt']));
z2_tbnn_pred = readmatrix(strcat(top_dir, ['Zone 2 TBNN output data\'...
    'Trials\Trial 1\Trial1_seed1_TBNN_test_prediction_bij.txt']));
z1_tkenn_pred = readmatrix(strcat(top_dir, ['Zone 1 TKENN output data\'...
    'Trials\Trial 1\Trial1_seed1_TKENN_test_prediction.txt']));
z2_tkenn_pred = readmatrix(strcat(top_dir, ['Zone 2 TKENN output data\'...
    'Trials\Trial 1\Trial1_seed1_TKENN_test_prediction.txt']));

% Find split idx for the zonal predictions ✓
z1_split_idx = find_split_idx(zrecon_obj, z1_tbnn_pred); % ✓ 54949
z1_split_idx_check = find_split_idx(zrecon_obj, z1_tkenn_pred); % ✓ 54949
z2_split_idx = find_split_idx(zrecon_obj, z2_tbnn_pred); % ✓ 25347
z2_split_idx_check = find_split_idx(zrecon_obj, z2_tkenn_pred); % ✓ 25347
assert(z1_split_idx == z1_split_idx_check)
assert(z2_split_idx == z2_split_idx_check)

% Convert zonal predictions to case predictions ✓
[U2_tbnn_pred, U4_tbnn_pred] = zrecon_obj.FBFS5_split_and_cat(...
    z1_tbnn_pred, z1_split_idx, z2_tbnn_pred, z2_split_idx, num_rows); % ✓
[U2_tkenn_pred, U4_tkenn_pred] = zrecon_obj.FBFS5_split_and_cat(...
    z1_tkenn_pred, z1_split_idx, z2_tkenn_pred, z2_split_idx, num_rows); % ✓

%% Write case prediction results ✓✓

% Sort rows in (Cx, Cy) order ✓
U2_tbnn_pred = sortrows(U2_tbnn_pred, [1, 2]);
U4_tbnn_pred = sortrows(U4_tbnn_pred, [1, 2]);
U2_tkenn_pred = sortrows(U2_tkenn_pred, [1, 2]);
U4_tkenn_pred = sortrows(U4_tkenn_pred, [1, 2]);

% Write results to file ✓
% zrecon_obj.write_tbnn_reconst(U2_tbnn_pred, ...
%     'U2_tbnn_zonal_pred_reconstructed.txt') % ✓
% zrecon_obj.write_tbnn_reconst(U4_tbnn_pred, ...
%     'U4_tbnn_zonal_pred_reconstructed.txt') % ✓
zrecon_obj.write_tkenn_reconst(U2_tkenn_pred, ...
    'U2_tkenn_zonal_pred_reconstructed.txt') % ✓
zrecon_obj.write_tkenn_reconst(U4_tkenn_pred, ...
    'U4_tkenn_zonal_pred_reconstructed.txt') % ✓
