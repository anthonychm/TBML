classdef CFD_data_reader_class
    % An instance of this class is a reader of CFD data for an OpenFOAM 
    % case directory
    
    properties
        dir {mustBeText} = 'dummy'
    end
    
    methods
        function [Cx, Cy, Cz, k, eps, gradU, gradp, U, gradk, tauij] = ...
                read_RANS_data(obj, pressure_tf, tke_tf) % ✓
            % Extract raw .txt data from a RANS case
            
            Cx = readmatrix(strcat(obj.dir, 'Cx.txt'));
            Cy = readmatrix(strcat(obj.dir, 'Cy.txt'));
            Cz = readmatrix(strcat(obj.dir, 'Cz.txt'));
            
            k = readmatrix(strcat(obj.dir, 'k.txt'));
            eps = readmatrix(strcat(obj.dir, 'eps.txt'));
            gradU = readmatrix(strcat(obj.dir, 'gradU.txt'));
            
            if pressure_tf == true
                gradp = readmatrix(strcat(obj.dir, 'gradp.txt'));
                U = readmatrix(strcat(obj.dir, 'U.txt'));
            end
            
            if tke_tf == true
                gradk = readmatrix(strcat(obj.dir, 'gradk.txt'));
            end
            
            tauij = readmatrix(strcat(obj.dir, 'tauij.txt'));
        end
        
        function [outlet_k, outlet_eps, outlet_U, outlet_tauij] = ...
                read_RANS_outlet_data(obj, pressure_tf) % ✓
            % Extract raw outlet .csv data
            
            outlet_k_and_eps = readmatrix(strcat(obj.dir, ...
                'outlet_k_and_eps.csv'));
            outlet_k = outlet_k_and_eps(:, 1);
            outlet_eps = outlet_k_and_eps(:, 2);
            
            if pressure_tf == true
                outlet_U = readmatrix(strcat(obj.dir, 'outlet_U.csv'));
            end
            
            outlet_tauij = readmatrix(strcat(obj.dir, 'outlet_tauij.csv'));
        end
        
        function [Cx, Cy, Cz, UPrime2Mean] = read_LES_data(obj) % ✓
            % Extract raw .txt data from an LES case
            
            Cx = readmatrix(strcat(obj.dir, 'Cx.txt'));
            Cy = readmatrix(strcat(obj.dir, 'Cy.txt'));
            Cz = readmatrix(strcat(obj.dir, 'Cz.txt'));
            UPrime2Mean = readmatrix(strcat(obj.dir, 'UPrime2Mean.txt'));
        end
    end
        
    methods (Static)
        function [inlet_k, inlet_eps] = five_fbfs_RANS_inlet_dict(...
                inlet_vel, num_inlet) % ✓
            % Define inlet variables dictionary for FBFS5
            
            keys = {'1', '2', '2.5', '3', '4'};
            inlet_k_vals = [0.00015, 0.0006, 0.0009375, 0.00135, 0.0024];
            inlet_eps_vals = [3.942e-05, 0.00031482, 0.00061511, ...
                0.0010629, 0.00251942];
            inlet_k_dict = containers.Map(keys, inlet_k_vals);
            inlet_eps_dict = containers.Map(keys, inlet_eps_vals);
            
            inlet_k = repmat(inlet_k_dict(inlet_vel), num_inlet, 1);
            inlet_eps = repmat(inlet_eps_dict(inlet_vel), num_inlet, 1);
        end
    end
end

