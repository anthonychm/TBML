classdef FBFS5_plotter_class
    % This class contains methods for FBFS5 a priori plots

    methods (Static)
        function plot_data = restore_nan_block(plot_data, plot_data_cp) % ✓
            % Restore nans represening solid block
            for i = 1:size(plot_data_cp, 1)
                for j = 1:size(plot_data_cp, 2)
                    if isnan(plot_data_cp(i, j))
                        plot_data(i, j) = NaN;
                    end
                end
            end
        end
        
        function plot_data = smooth_data(plot_data, method, window) % ✓
            % Smooth plot data
            % Ensure plot_data is in mesh format
            plot_data_cp = plot_data;
            plot_data = smoothdata(plot_data, method, window);
            plot_data = FBFS5_plotter_class.restore_nan_block(...
                plot_data, plot_data_cp); % ✓
        end
        
        function plot_data = clip_colors(plot_data, cmin, cmax) % ✓
            % Clip the colours in the contour plots
            plot_data_cp = plot_data;
            plot_data = max(min(plot_data, cmax), cmin);
            plot_data = FBFS5_plotter_class.restore_nan_block(...
                plot_data, plot_data_cp); % ✓
        end
        
        function i = contour_rep(i, mesh_x, mesh_y, plot_data, ...
                num_mesh_rows, cmin, cmax, num_colours, title_text) % ✓
            % Repeated code in contour plotting
            i = i + 1;
            lower = (i*num_mesh_rows) - num_mesh_rows + 1;
            upper = i*num_mesh_rows;
            [~, c] = contourf(mesh_x, mesh_y, plot_data(lower:upper, :), ...
                linspace(cmin, cmax, num_colours + 1));
            c.LineStyle = 'none';
            ax = gca;
            ax.FontSize = 24;
            xticks(0:2:22)
            ylabel('y / h_s', 'FontSize', 26)
            title(title_text)
        end
        
        function contour(mesh_x, mesh_y, plot_data, num_colours, ...
                plot_var, plot_second_tile) % ✓
            
            % Create contour plots ✓
            num_mesh_rows = size(mesh_x, 1);
            tiledlayout(size(plot_data, 1)/num_mesh_rows, 1);
            cmin = 0; %min(min(plot_data));
            cmax = 3; %max(max(plot_data));
            plot_data = FBFS5_plotter_class.clip_colors(plot_data, ...
                cmin, cmax); % ✓
            i = 0;
            
            % RANS tile
            tile_1 = nexttile;
            i = FBFS5_plotter_class.contour_rep(i, mesh_x, mesh_y, ...
                plot_data, num_mesh_rows, cmin, cmax, num_colours, ...
                strcat('RANS ', plot_var)); % ✓
            
            % LES tile
            tile_2 = nexttile;
            i = FBFS5_plotter_class.contour_rep(i, mesh_x, mesh_y, ...
                plot_data, num_mesh_rows, cmin, cmax, num_colours, ...
                strcat('LES ', plot_var)); % ✓
            
            % One-zone tile
            tile_3 = nexttile;
            i = FBFS5_plotter_class.contour_rep(i, mesh_x, mesh_y, ...
                plot_data, num_mesh_rows, cmin, cmax, num_colours, ...
                strcat('First Pred ', plot_var)); % ✓
            
            % Format colormap axis ✓
            colormap(jet(num_colours))
            caxis(tile_1, [cmin, cmax])
            caxis(tile_2, [cmin, cmax])
            caxis(tile_3, [cmin, cmax])
            
            % Zonal tile ✓
            if plot_second_tile == false
                linkaxes([tile_1 tile_2 tile_3], 'xy')
            else
                tile_4 = nexttile;
                i = FBFS5_plotter_class.contour_rep(i, mesh_x, mesh_y, ...
                    plot_data, num_mesh_rows, cmin, cmax, num_colours, ...
                    strcat('Second pred ', plot_var)); % ✓
                linkaxes([tile_1 tile_2 tile_3 tile_4], 'xy')
                caxis(tile_4, [cmin, cmax])
            end
            
            % Format x and y axes ✓
            xlabel('x / h_s', 'FontSize', 26)
            xlim(tile_1, [min(min(mesh_x)), max(max(mesh_x))]);
            ylim(tile_1, [0 2]);
            
            % Format colorbar, previously (num_colours + 1) ✓
            c = colorbar('southoutside', 'Ticks', ...
                linspace(cmin, cmax, 11));
            tick_labels = arrayfun(@(x) sprintf('%.2f', x), ...
                linspace(cmin, cmax, 11), 'un', 0);
            set(c, 'TickLabels', tick_labels);
            c.FontSize = 12; %12, 24
        end
        
        function data = extract_from_ref(sim_type, var, lower, upper)
            % Extract bij, k or tauij from a ref table
            ref_table = readmatrix(strcat(sim_type, '_ref_table.txt'));
            if strcmp(var, 'bij')
                data = ref_table(lower:upper, ...
                    (num_dims + 1):(num_dims + 9));
            elseif strcmp(var, 'k')
                data = ref_table(lower:upper, (num_dims + 10));
            elseif strcmp(var, 'tauij')
                data = ref_table(lower:upper, ...
                    (num_dims + 11):(num_dims + 20));
            else
                error('invalid error variable');
            end
        end
        
        function error_contour(mesh_x, mesh_y, error_plot_data, num_colours)
            % Create contour tiles (3x3)
            tiledlayout(3, 3);
            dict_obj = dict_class;
            
            % Loop through all 9 tiles
            for i = 1:9
                % Plot error contour
                nexttile
                cmin = min(min(error_plot_data(:, :, i)));
                cmax = max(max(error_plot_data(:, :, i)));
                contourf(mesh_x, mesh_y, error_plot_data(:, :, i), ...
                    linspace(cmin, cmax, num_colours + 1))
                xlim([0 70/3]);
                ylim([0 2]);
                
                ax = gca;
                ax.FontSize = 12;
                ylabel('y / h_s', 'FontSize', 14)
                xlabel('x / h_s', 'FontSize', 14)
                
                % Extract comp values from dict class
                [comp1, comp2] = dict_obj.Re_comp_dict_reverse(i);
                title(strcat('\tau_', num2str(comp1), '_', ...
                    num2str(comp2), ' error'))
                
                % Show colorbar
                colormap(parula(num_colours))
                caxis([cmin, cmax])
                c = colorbar('eastoutside', 'Ticks', ...
                    linspace(cmin, cmax, num_colours + 1));
                tick_labels = arrayfun(@(x) sprintf('%.2f', x), ...
                    linspace(cmin, cmax, num_colours + 1), 'un', 0);
                set(c, 'TickLabels', tick_labels);
                c.FontSize = 12;
            end
        end
        
        function closest_val = find_closest_val(query, base_vector)
            % Find the closest value in a vector to a query value
            [~, ~, idx] = unique(abs(base_vector - query));
            closest_val = base_vector(idx == 1);
            closest_val = closest_val(1, 1);
        end
    end
end

