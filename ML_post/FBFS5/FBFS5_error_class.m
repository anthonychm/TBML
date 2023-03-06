classdef FBFS5_error_class
    % This class contains methods for calculating MSE and RMSE from FBFS5
    % predictions
    
    methods (Static)
        function base_data = extract_base_data(base_sim, ...
                RANS_plot_data, LES_plot_data)
            % Extract base data for error calculation
            if strcmp(base_sim, 'RANS')
                base_data = RANS_plot_data;
            elseif strcmp(base_sim, 'LES')
                base_data = LES_plot_data;
            else
                error('invalid base data')
            end
        end
        
        function [Cx, Cy] = extract_Cx_Cy(rmse_region, inlet_vel, top_dir)
            % Extract Cx and Cy coordinates for zonal error calculation
            if strcmp(rmse_region, 'zone1')
                z1 = readmatrix(strcat(top_dir, ['Code\TBNN_workflow\'...
                    'Driver\Zonal high opt pfalse kfalse\'...
                    'Zone 1 TBNN output data\Trials\Trial 1\'...
                    'Trial1_seed1_TBNN_test_prediction_bij.txt']));
                if inlet_vel == 2
                    Cx = z1(1:54949, 1); % From zonal reconstructor
                    Cy = z1(1:54949, 2); % From zonal reconstructor
                elseif inlet_vel == 4
                    Cx = z1(54950:end, 1); % From zonal reconstructor
                    Cy = z1(54950:end, 2); % From zonal reconstructor
                else
                    error('invalid inlet velocity')
                end
            elseif strcmp(rmse_region, 'zone2')
                z2 = readmatrix(strcat(top_dir, ['Code\TBNN_workflow\'...
                    'Driver\Zonal high opt pfalse kfalse\'...
                    'Zone 2 TBNN output data\Trials\Trial 1\'...
                    'Trial1_seed1_TBNN_test_prediction_bij.txt']));
                if inlet_vel == 2
                    Cx = z2(1:25347, 1); % From zonal reconstructor
                    Cy = z2(1:25347, 2); % From zonal reconstructor
                elseif inlet_vel == 4
                    Cx = z2(25348:end, 1); % From zonal reconstructor
                    Cy = z2(25348:end, 2); % From zonal reconstructor
                else
                    error('invalid inlet velocity')
                end
            else
                error('invalid rmse region')
            end
        end
    end
end

