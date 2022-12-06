%% PREAMBLE %%
% This script is the successor of RANS_reader_FBFS_centreplane.m and the 
% var_reader_RANS.m scripts in channel flow CFD postprocessing.

% This script reads RANS CFD data used for calculating Sij, Rij, Ak and Ap
% for TBNN inputs. In addition, input scalar markers and zonal markers can
% also be included. Reynolds stresses are included by default. Centreplane 
% results of these quantities are extracted and saved as a .mat array.

%% Read CFD results ✓✓

clear variables
top_dir = ['C:\Users\Antho\Dropbox (The University of Manchester)\'...
    'PhD_Anthony_Man\'];
addpath(strcat(top_dir, 'Code\GoTo'), '..\Markers', '..\');

% Define objects ✓
reader_obj = CFD_data_reader_class;
extractor_obj = cplane_extractor_class;
writer_obj = write_class;
reformat_obj = reformat_class;

% ***** PARAMETERS ***** % ✓
inlet_vel = 4;

gradU_tf = true;
pressure_tf = true;
tke_tf = true;
imarkers_tf = false;
num_M = 0;
zmarkers_tf = true;
num_Z = 1;
% ***** PARAMETERS ***** %

if imarkers_tf == true
    pressure_tf = true;
    tke_tf = true;
end

% Constants ✓
num_inlet = 5000;
num_wall_zeros_ref = 91800;
num_unique_cplane_coords = 82136;
nu = 1e-5;

% Read RANS FBFS CFD data ✓
inlet_vel = num2str(inlet_vel);
reader_obj.dir = strcat(top_dir, 'HPC\CFD\FBFS\RANS\solidBlockU', ...
    inlet_vel, '\solidBlockU', inlet_vel, '_');
gradk = NaN;
[Cx, Cy, Cz, k, eps, gradU, gradp, U, gradk, tauij] = read_RANS_data(...
    reader_obj, pressure_tf, tke_tf); % ✓

disp('RANS data reading complete')

%% Extract inlet and outlet results ✓✓

% Inlet k, eps and U ✓
[inlet_k, inlet_eps] = reader_obj.five_fbfs_RANS_inlet_dict(inlet_vel, ...
    num_inlet); % ✓
if pressure_tf == true
    inlet_U = repmat([str2double(inlet_vel), 0, 0], num_inlet, 1);
end

% Outlet k, eps, U and tauij ✓
[outlet_k, outlet_eps, outlet_U, outlet_tauij] = read_RANS_outlet_data(...
    reader_obj, pressure_tf); % ✓

% Reformat tauij cols from: u2, v2, w2, uv, uw, vw 
%                       to: u2, uv, uw, v2, vw, w2 ✓
outlet_tauij = reformat_class.reorder_symmTensor(outlet_tauij); % ✓

%% Include inlet, outlet and at-wall values ✓✓

% Concatenate inlet, outlet and at-wall values to existing variable arrays
assert(size(k, 1) == size(eps, 1))
assert(size(k, 1) == size(U, 1))

% Inlet tauij values already provided by OpenFOAM ✓
assert(size(k, 1) == size(tauij, 1) - num_inlet)

k = cat(1, k, inlet_k, outlet_k);
num_wall_zeros = height(Cx) - height(k);
assert(num_wall_zeros == num_wall_zeros_ref)

k = cat(1, k, zeros(num_wall_zeros, 1));
eps = cat(1, eps, inlet_eps, outlet_eps, zeros(num_wall_zeros, 1));

if pressure_tf == true
    U = cat(1, U, inlet_U, outlet_U, zeros(num_wall_zeros, 3));
end

tauij = cat(1, tauij, outlet_tauij, zeros(num_wall_zeros, 6));

disp('Inlet, outlet and at-wall values included')

%% Reformat gradU, calculate Sij and Rij, and combine arrays ✓✓

% Reformat gradU from [dU/dx, dV/dx, dW/dx ...] to [dU/dx, dU/dy, dU/dz...]
% ✓
gradU = reformat_obj.transpose_gradU_flat(gradU); % ✓

% Calculate Sij and Rij if using input markers [NEEDS CHECK]
if imarkers_tf == true
    calc_obj = calc_class;
    Sij = calc_obj.Sij_calc(gradU); % ✓
    Rij = calc_obj.Omegaij_calc(gradU); % ✓
end

% Combine (k, eps and gradU)for centreplane extraction ✓
k_eps_and_gradU = cat(2, k, eps, gradU);

if pressure_tf == true
    % Combine (gradp and U) for centreplane extraction ✓
    gradp_and_U = cat(2, gradp, U);
elseif pressure_tf == false
    gradp_and_U = NaN;
end

%% Calculate input markers M and zonal markers Z if required ✓✓

markers_obj = Markers_class;

% Input markers [NEEDS CHECK]
if imarkers_tf == true
    Qcrit = markers_obj.calc_Qcrit(Sij, Rij);
    Ti = markers_obj.calc_Ti(k, U);
    dists_array = FBFS_calc_d_func(Cx, Cy);
    Ret = markers_obj.calc_Ret(k, dists_array(:, end), nu);
    gradp_sl = markers_obj.calc_gradp_sl(U, gradp);
    timescale_ratio = markers_obj.calc_timescale_ratio(Sij, k, eps);
    conv_prod_ratio = markers_obj.calc_tke_conv_prod_ratio(U, gradk, ...
        tauij, Sij);
    Re_stress_ratio = markers_obj.calc_Re_stress_ratio(tauij, k);
    M = cat(2, Qcrit, Ti, Ret, grap_sl, timescale_ratio, ...
        conv_prod_ratio, Re_stress_ratio);
elseif imarkers_tf == false
    M = NaN;
end

% Zonal markers ✓
if zmarkers_tf == true
    nondim_k = markers_obj.calc_nondim_k(k, str2double(inlet_vel)); % ✓
    Z = nondim_k;
elseif zmarkers_tf == false
    Z = NaN;
end

disp('Markers calculated')

%% Extract and write centreplane results ✓✓

% Define extractor object attributes ✓
extractor_obj.Cx = Cx;
extractor_obj.Cy = Cy;
extractor_obj.Cz = Cz;

% Get list of cell indexes corresponding to (Cx, Cy) coordinates at
% centreplane ✓
unique_coords = find_unique_Cx_Cy(extractor_obj, num_unique_cplane_coords); % ✓
closest_Cz = find_closest_Cz(extractor_obj); % ✓
cell_idx_list = extract_cplane_idx(extractor_obj, unique_coords, ...
    closest_Cz); % ✓

% Assemble centreplane results array ✓
[cplane_results, cols_dict] = extractor_obj.assemble_cplane_results(...
    cell_idx_list, unique_coords, Cx, Cy, k_eps_and_gradU, ...
    gradp_and_U, gradk, M, Z, tauij, gradU_tf, pressure_tf, tke_tf, ...
    imarkers_tf, zmarkers_tf, num_M, num_Z); % ✓

% Write centreplane results array ✓
num_dims = 2;
results = sortrows(cplane_results, 1:num_dims);
table_name = strcat('solidBlockU', inlet_vel, '_RANS_cplane_table');
writer_obj.write_case_results(results, unique_coords, cols_dict, ...
    num_dims, num_M, num_Z, gradU_tf, pressure_tf, tke_tf, table_name) % ✓

disp('RANS centreplane results file written')
