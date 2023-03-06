classdef zonal_reconstruct_class
    % This class contains methods for zonal reconstruction of prediction
    % results.
    
    properties
        num_splits {mustBeInteger} = 0
    end
    
    methods
        function z_split_idx = find_split_idx(obj, zonal_pred) % ✓
            % Find the idx where case splits occur
            zonal_Cx = zonal_pred(:, 1);
            z_split_idx = find(diff(zonal_Cx) < 0 , obj.num_splits); % Check position!
        end
    end
        
    methods (Static)
        function [U2_reconst, U4_reconst] = FBFS5_split_and_cat(...
                z1_pred, z1_split_idx, z2_pred, z2_split_idx, num_rows) % ✓
            % Convert zonal predictions to case predictions
            U2_reconst = z1_pred(1:z1_split_idx, :);
            U2_reconst = cat(1, U2_reconst, z2_pred(1:z2_split_idx, :));
            assert(size(U2_reconst, 1) == num_rows)
            U4_reconst = z1_pred((z1_split_idx + 1):end, :);
            U4_reconst = cat(1, U4_reconst, ...
                z2_pred((z2_split_idx + 1):end, :));
            assert(size(U4_reconst, 1) == num_rows)
        end
        
        function write_tbnn_reconst(tbnn_reconst, table_name) % ✓
            % Write reconstructed zonal tbnn results
            header = {'Cx', 'Cy', 'b11', 'b12', 'b13', 'b21', 'b22', ...
                'b23', 'b31', 'b32', 'b33'};
            tbnn_reconst = array2table(tbnn_reconst, 'VariableNames', ...
                header);
            writetable(tbnn_reconst, table_name, 'Delimiter', ' ')
        end
        
        function write_tkenn_reconst(tkenn_reconst, table_name) % ✓
            % Write reconstructed zonal tkenn results
            header = {'Cx', 'Cy', 'log(k)', 'k'};
            tkenn_reconst = array2table(tkenn_reconst, 'VariableNames', ...
                header);
            writetable(tkenn_reconst, table_name, 'Delimiter', ' ')
        end
    end
end

