classdef bary_map_class
    % This class contains methods for creating a barycentric map with bij
    % data in mesh format

    methods (Static)
        function bij_mesh_line = extract_mesh_line(inlet_vel, source, ...
                folder, bij_comp, sample_dir, start_idx, end_idx, const_idx) % ✓
            % Load single bij comp in mesh format
            bij_mesh_data = struct2cell(load(strcat(folder, 'U', ...
                num2str(inlet_vel), '_', source, '_', bij_comp, '_mesh.mat')));
            bij_mesh_data = bij_mesh_data{1};
            
            % Extract line data from mesh
            if strcmp(sample_dir, 'x')
                bij_mesh_line = bij_mesh_data(const_idx, start_idx:end_idx);
            elseif strcmp(sample_dir, 'y')
                bij_mesh_line = bij_mesh_data(start_idx:end_idx, const_idx);
            end
        end
        
        function [b11, b12, b13, b21, b22, b23, b31, b32, b33] = ...
                load_mesh_line(inlet_vel, source, folder, sample_dir, ...
                start_idx, end_idx, const_idx) % ✓
            % Extract line data from mesh for all bij components
            b11 = bary_map_class.extract_mesh_line(inlet_vel, source, ...
                folder, 'b11', sample_dir, start_idx, end_idx, const_idx); % ✓
            b12 = bary_map_class.extract_mesh_line(inlet_vel, source, ...
                folder, 'b12', sample_dir, start_idx, end_idx, const_idx); % ✓
            b13 = bary_map_class.extract_mesh_line(inlet_vel, source, ...
                folder, 'b13', sample_dir, start_idx, end_idx, const_idx); % ✓
            b21 = bary_map_class.extract_mesh_line(inlet_vel, source, ...
                folder, 'b21', sample_dir, start_idx, end_idx, const_idx); % ✓
            b22 = bary_map_class.extract_mesh_line(inlet_vel, source, ...
                folder, 'b22', sample_dir, start_idx, end_idx, const_idx); % ✓
            b23 = bary_map_class.extract_mesh_line(inlet_vel, source, ...
                folder, 'b23', sample_dir, start_idx, end_idx, const_idx); % ✓
            b31 = bary_map_class.extract_mesh_line(inlet_vel, source, ...
                folder, 'b31', sample_dir, start_idx, end_idx, const_idx); % ✓
            b32 = bary_map_class.extract_mesh_line(inlet_vel, source, ...
                folder, 'b32', sample_dir, start_idx, end_idx, const_idx); % ✓
            b33 = bary_map_class.extract_mesh_line(inlet_vel, source, ...
                folder, 'b33', sample_dir, start_idx, end_idx, const_idx); % ✓

            assert(max(sum(b12 - b21)) < 1e-12)
            assert(max(sum(b13 - b31)) < 1e-12)
            assert(max(sum(b23 - b32)) < 1e-12)
        end
        
        function bij = reformat_three_by_three(b11, b12, b13, b21, b22, ...
                b23, b31, b32, b33) % ✓
            % Reformat mesh line bij data into 3 x 3 x num points format
            bij = NaN(3, 3, sum(~isnan(b11), 'all'));
            count = 0;
            for i = 1:size(b11, 1)
                for j = 1:size(b11, 2)
                    if ~isnan(b11(i, j))
                        count = count + 1;
                        bij(1, 1, count) = b11(i, j);
                        bij(1, 2, count) = b12(i, j);
                        bij(1, 3, count) = b13(i, j);
                        bij(2, 1, count) = b21(i, j);
                        bij(2, 2, count) = b22(i, j);
                        bij(2, 3, count) = b23(i, j);
                        bij(3, 1, count) = b31(i, j);
                        bij(3, 2, count) = b32(i, j);
                        bij(3, 3, count) = b33(i, j);
                    end
                end
            end

            bij_slice_example = bij(:, :, 1);
            assert(sum(isnan(bij), 'all') == 0)
        end
        
        function [C1, C2, C3] = calc_coeffs(lambda) % ✓
            % Calculate C coeffs
            C1 = lambda(1, 1) - lambda(2, 2);
            C2 = 2*(lambda(2, 2) - lambda(3, 3));
            C3 = (3*lambda(3, 3)) + 1;
            assert(0.8 < (C1 + C2 + C3))
            assert((C1 + C2 + C3) < 1.2)
        end
        
        function create_tri(A_x, A_y, B_x, B_y, C_x, C_y) % ✓
            % Create equilateral triangle background for Barycentric map
            v = [A_x A_y; B_x B_y; C_x C_y];
            f = [1 2 3];
            c = [1 0 0; 0 1 0; 0 0 1];
            figure
            patch('Faces', f, 'Vertices', v, 'FaceVertexCData', c, ...
                'FaceColor', 'interp');
            %set(gcf, 'Position',  [100, 100, 200, 200])
            set(gcf, 'Position',  [100, 100, 400, 400])
            hold on
        end
        
        function [x_bary, y_bary] = calc_coords(C1, A_x, A_y, ...
                C2, B_x, B_y, C3, C_x, C_y) % ✓
            % Calculate Barycentric map coordinates
            x_bary = (C1*A_x) + (C2*B_x) + (C3*C_x);
            y_bary = (C1*A_y) + (C2*B_y) + (C3*C_y);
        end
        
        function flag = find_out_points(x_bary, y_bary, A_x, A_y, ...
                B_x, B_y, C_x, C_y) % ✓
            % Find points outside of the Barycentric map
            if y_bary < 0
                flag = true;
            elseif y_bary > ((C_y - B_y)/(C_x - B_x))*x_bary
                flag = true;
            elseif y_bary > (((A_y - C_y)/(A_x - C_x))*x_bary)...
                    -((A_y - C_y)/(A_x - C_x))
                flag = true;
            else
                flag = false;
            end
        end
        
        function [coords, flag_list] = plot_bary_coords(bij, A_x, A_y, B_x, B_y, ...
                C_x, C_y, marker, save_coords) % ✓
            % Calculate all x_bary and y_bary coordinates for a mesh line
            % in 3 x 3 format
            flag_list = [];
            coords = [];
            
            for i = 1:size(bij, 3)
                [~, lambda] = eig(bij(:, :, i));
                [~, idx] = sort(diag(lambda), 'descend');
                lambda = lambda(idx, idx);
                [C1, C2, C3] = bary_map_class.calc_coeffs(lambda); % ✓
                [x_bary, y_bary] = bary_map_class.calc_coords(...
                    C1, A_x, A_y, C2, B_x, B_y, C3, C_x, C_y); % ✓
                
                if save_coords == true
                    coords = cat(1, coords, [x_bary y_bary]);
                end
                
                flag = bary_map_class.find_out_points(x_bary, y_bary, ...
                    A_x, A_y, B_x, B_y, C_x, C_y); % ✓
                if flag == true
                    flag_list = cat(1, flag_list, i);
                end
                %plot(x_bary, y_bary, marker, 'MarkerSize', 5)
                plot(x_bary, y_bary, marker, 'MarkerSize', 2+(i*0.15))
            end
        end
        
        function mean_coords = calc_mean_coords(coords)
            % Calculate mean barycentric coords
            mean_x_bary = mean(coords(:, 1));
            mean_y_bary = mean(coords(:, 2));
            mean_coords = cat(2, mean_x_bary, mean_y_bary);
        end
        
        function mean_unit_vec = calc_mean_unit_vec(coords)
            % Calculate mean unit vector between barycentric coords
            unit_vecs = [];
            for i = 2:size(coords, 1)
                vec = coords(i, :) - coords(i-1, :);
                unit_vec = vec/sqrt(vec(1, 1).^2 + vec(1, 2).^2);
                unit_vecs = cat(1, unit_vecs, unit_vec);
            end
            mean_unit_vec = [mean(unit_vecs(:, 1)), mean(unit_vecs(:, 2))];
        end
        
        function arrow = incl_arrow(mean_coords, mean_unit_vec)
            % Calculate the start and end coordinates of an arrow, given a
            % centrepoint and vector and create the arrow
            % ** Not used in Oct 2022 PoF paper **
            start_coords = [mean_coords(1) - (mean_unit_vec(1)/16), ...
                mean_coords(2) - (mean_unit_vec(2)/16)];
            arrow = annotation('textarrow');
            set(arrow, 'parent', gca, 'position', [start_coords, mean_unit_vec/8])
        end
    end
end

