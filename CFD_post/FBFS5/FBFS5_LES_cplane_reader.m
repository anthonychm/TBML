%% PREAMBLE %%
% This script is the successor of LES_reader_FBFS_centreplane.m and  
% UPrime2Mean_reader_LES.m in channel flow CFD postprocessing.

% This script reads LES UPrime2Mean data for use as TBNN ground truth 
% output. Centreplane results of UPrime2Mean are extracted and saved as a 
% .mat array.

%% Read CFD results ✓✓

clear variables
top_dir = ['C:\Users\Antho\Dropbox (The University of Manchester)\'...
    'PhD_Anthony_Man\'];
addpath(strcat(top_dir, 'Code\GoTo'), '..\');

% Define objects ✓
reader_obj = CFD_data_reader_class;
extractor_obj = cplane_extractor_class;
writer_obj = write_class;

% ***** PARAMETERS ***** % ✓
inlet_vel = 4;
% ***** PARAMETERS ***** %

% Constants ✓
num_wall_zeros_ref = 216000;
num_unique_cplane_coords = 106800;

% Read LES FBFS CFD data ✓
if ismember(inlet_vel, [1, 2, 4])
    inlet_vel = num2str(inlet_vel);
    reader_obj.dir = strcat(top_dir, 'HPC\CFD\FBFS\LES_MJ\HTsolidBlockLESU', ...
        inlet_vel, '_FINAL\solidBlockU', inlet_vel, '_LES_');
else
    inlet_vel = num2str(inlet_vel);
    reader_obj.dir = strcat(top_dir, 'HPC\CFD\FBFS\LES\solidBlockU', ...
        inlet_vel, '\solidBlockU', inlet_vel, '_LES_');
end

[Cx, Cy, Cz, UPrime2Mean] = read_LES_data(reader_obj); % ✓
assert(height(Cx) == height(UPrime2Mean) + num_wall_zeros_ref)

disp('LES data reading complete')

%% Include at-wall values ✓✓

num_wall_zeros = height(Cx) - height(UPrime2Mean);
assert(num_wall_zeros == num_wall_zeros_ref)
UPrime2Mean = cat(1, UPrime2Mean, zeros(num_wall_zeros, 6));

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
    cell_idx_list, unique_coords, Cx, Cy, NaN, NaN, NaN, NaN, NaN, ...
    UPrime2Mean, false, false, false, false, false, 0, 0); % ✓

% Write centreplane results array ✓
num_dims = 2;
results = sortrows(cplane_results, 1:num_dims);
table_name = strcat('solidBlockU', inlet_vel, '_LES_cplane_table');
writer_obj.write_case_results(results, unique_coords, cols_dict, ...
    num_dims, 0, 0, false, false, false, table_name) % ✓

disp('LES centreplane results file written')
