classdef cplane_extractor_class
    % An instance of this class is a cplane data extractor for any OpenFOAM
    % CFD case
    
    properties
        Cx {mustBeVector} = NaN(1, 2)
        Cy {mustBeVector} = NaN(1, 2)
        Cz {mustBeVector} = NaN(1, 2)
    end
    
    methods
        function unique_coords = find_unique_Cx_Cy(obj, target) % ✓
            % Find unique Cx, Cy coordinate pairs 
            coords = cat(2, obj.Cx, obj.Cy);
            unique_coords = unique(coords, 'rows');
            assert(size(unique_coords, 1) == target)
        end
        
        function closest_Cz = find_closest_Cz(obj) % ✓
            % Find value of Cz closest to centreplane z
            cplane_z = min(obj.Cz) + ((max(obj.Cz) - min(obj.Cz))/2);
            unique_z = unique(obj.Cz);
            [~, ~, i] = unique(abs(unique_z - cplane_z));
            closest_Cz = unique_z(i == 1);
            closest_Cz = closest_Cz(1, 1);
        end
        
        function cell_idx_list = extract_cplane_idx(obj, unique_coords, closest_Cz) % ✓
            % Extract idx of cells with closest Cz
            cell_idx_list = zeros(size(unique_coords, 1)+1, 1);
            for i = 1:height(obj.Cz)
                if obj.Cz(i) == closest_Cz
                    next_zero = find(~cell_idx_list, 1, 'first');
                    cell_idx_list(next_zero) = i;
                end
            end
            
            assert(nnz(~cell_idx_list) == 1)
            cell_idx_list = cell_idx_list(any(cell_idx_list, 2));
            assert(size(cell_idx_list, 1) == size(unique_coords, 1));
        end
    end
    
    methods (Static)
        function [cplane_results, cols_dict] = assemble_cplane_results(cell_idx_list, ...
                unique_coords, Cx, Cy, k_eps_and_gradU, gradp_and_U, gradk, ...
                M, Z, tauij, gradU_tf, pressure_tf, tke_tf, imarkers_tf, zmarkers_tf, ...
                num_M, num_Z) % ✓
            
            % Extract centreplane coordinate points ✓
            num_rows = size(unique_coords, 1);
            cplane_results = NaN(num_rows, 2);
            for i = 1:height(cell_idx_list)
                cplane_results(i, 1) = Cx(cell_idx_list(i));
                cplane_results(i, 2) = Cy(cell_idx_list(i));
            end
            
            % Define dictionary that gives number of columns per additional
            % variable ✓
            keys = {'gradU', 'pressure', 'tke', 'input markers', 'zonal markers', 'tauij'};
            num_cols = [0, 0, 0, num_M, num_Z, 6];
            cols_dict = containers.Map(keys, num_cols);
            
            % Extract centreplane results for Sij and Rij ✓
            if gradU_tf == true
                cols_dict('gradU') = 11;
                cplane_results = cplane_extractor_class.extract_cplane_results('gradU', ...
                k_eps_and_gradU, num_rows, cols_dict, cell_idx_list, cplane_results); % ✓
            end
            
            % Extract centreplane results for Ap ✓
            if pressure_tf == true
                cols_dict('pressure') = 6;
                cplane_results = cplane_extractor_class.extract_cplane_results('pressure', ...
                gradp_and_U, num_rows, cols_dict, cell_idx_list, cplane_results); % ✓
            end
            
            % Extract centreplane results for Ak ✓
            if tke_tf == true
                cols_dict('tke') = 3;
                cplane_results = cplane_extractor_class.extract_cplane_results('tke', gradk, ...
                num_rows, cols_dict, cell_idx_list, cplane_results); % ✓
            end
            
            % Extract centreplane input markers results ✓
            if imarkers_tf == true
                cplane_results = cplane_extractor_class.extract_cplane_results('input markers', ...
                M, num_rows, cols_dict, cell_idx_list, cplane_results); % ✓
            end
            
            % Extract centreplane zonal markers results ✓
            if zmarkers_tf == true
                cplane_results = cplane_extractor_class.extract_cplane_results('zonal markers', ...
                Z, num_rows, cols_dict, cell_idx_list, cplane_results); % ✓
            end
            
            % Extract centreplane tauij results ✓
            cplane_results = cplane_extractor_class.extract_cplane_results('tauij', tauij, ...
                num_rows, cols_dict, cell_idx_list, cplane_results); % ✓
        end
        
        function cplane_results = extract_cplane_results(var_string, ...
                var, num_rows, cols_dict, cell_idx_list, cplane_results) % ✓
            
            % This method extracts cplane results of a given variable
            cplane_minor = NaN(num_rows, cols_dict(var_string));
            for i = 1:height(cell_idx_list)
                cplane_minor(i, :) = var(cell_idx_list(i), :);
            end
            cplane_results = cat(2, cplane_results, cplane_minor);
        end
    end
end

