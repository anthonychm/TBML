import numpy as np
import math
from scipy.spatial import Voronoi
import torch
import torch.nn as nn

import sys
sys.path.append('../TBNN')
import TBNN as tbnn


class TBMix(nn.Module):
    def __init__(self, seed, num_kernels, structure=None, weight_init="dummy",
                 weight_init_params="dummy", incl_t0_gen=False):  #
        super(TBMix, self).__init__()
        if structure is None:
            raise Exception("Network structure not defined")
        self.structure = structure
        self.seed = seed
        self.num_kernels = num_kernels
        self.weight_init = weight_init
        weight_init_params = weight_init_params.replace(", ", "=")
        self.weight_init_params = weight_init_params.split("=")
        self.incl_t0_gen = incl_t0_gen

        def retrieve_af_params(af_df, layer):  #
            params = af_df.loc[[layer]][layer]
            af_param_dict = dict(param.split("=") for param in params)

            for key, value in af_param_dict.items():
                if value == "True":
                    af_param_dict[key] = True
                elif value == "False":
                    af_param_dict[key] = False
                try:
                    af_param_dict[key] = float(value)
                except:
                    pass

            return af_param_dict

        # Create PyTorch neural network
        self.net = nn.Sequential()

        # Invariant input layer to first hidden layer
        layer = 0
        self.net.add_module("layer1", nn.Linear(self.structure.num_inputs,
                                                self.structure.num_hid_nodes[layer]))
        if self.structure.af_df is None:
            self.net.add_module("af1", getattr(nn, self.structure.af[layer])())
        else:
            af_param_dict = retrieve_af_params(self.structure.af_df, layer)  #
            self.net.add_module("af1",
                                getattr(nn, self.structure.af[layer])(**af_param_dict))

        # First hidden layer to subsequent hidden layers
        for layer in range(1, self.structure.num_hid_layers):
            self.net.add_module("layer" + str(layer + 1),
                                nn.Linear(self.structure.num_hid_nodes[layer - 1],
                                          self.structure.num_hid_nodes[layer]))
            if self.structure.af_df is None:
                self.net.add_module("af" + str(layer + 1),
                                    getattr(nn, self.structure.af[layer])())
            else:
                af_param_dict = retrieve_af_params(self.structure.af_df, layer)  #
                self.net.add_module("af" + str(layer + 1),
                                    getattr(nn, self.structure.af[layer])(**af_param_dict))

        # Final hidden layer outputs (pi, mu, sigma)
        self.z_pi = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)
        self.z_mu_g1 = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)
        self.z_mu_g2 = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)
        self.z_mu_g3 = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)
        self.z_sigma = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)

        # Further hidden layer outputs if 3D tensor basis is used
        if self.structure.num_tensor_basis == 10:
            self.z_mu_g4 = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)
            self.z_mu_g5 = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)
            self.z_mu_g6 = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)
            self.z_mu_g7 = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)
            self.z_mu_g8 = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)
            self.z_mu_g9 = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)
            self.z_mu_g10 = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)

        # Include hidden layer outputs for t0_gen if required
        if self.incl_t0_gen is True:
            self.z_mu_g01 = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)
            self.z_mu_g02 = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)
            self.z_mu_g03 = nn.Linear(self.structure.num_hid_nodes[-1], self.num_kernels)

        print("Building of neural network complete")

        # Create initialization arguments dictionary and initialize weights and biases
        param_keys, param_val = [], []
        for i, string in enumerate(self.weight_init_params):
            if (i+2) % 2 == 0:
                param_keys.append(string)
            else:
                try:
                    param_val.append(float(string))
                except:
                    param_val.append(string)

        weight_init_dict = dict(zip(param_keys, param_val))

        def init_w_and_b(m):
            if isinstance(m, nn.Linear):
                init = getattr(nn.init, self.weight_init)
                init(m.weight, **weight_init_dict)
                m.bias.data.fill_(0.01)  # set initial bias value here

        self.net.apply(init_w_and_b)
        print("Weights and biases initialized")

    # Forward propagation
    def forward(self, x, tb):

        # Forward pass hidden network structure to obtain network outputs
        z_h = self.net(x)  # Tensor (batch size, num nodes in last hidden layer)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)  # Tensor (batch size, num kernels)
        mu_g1 = self.z_mu_g1(z_h)  # Tensor (batch size, num kernels)
        mu_g2 = self.z_mu_g2(z_h)  # Tensor (batch size, num kernels)
        mu_g3 = self.z_mu_g3(z_h)  # Tensor (batch size, num kernels)
        # sigma = torch.exp(self.z_sigma(z_h))  # Tensor (batch size, num kernels)
        sigma = nn.ELU()(self.z_sigma(z_h)) + 1 + 1e-15

        if self.structure.num_tensor_basis == 10:
            mu_g4 = self.z_mu_g4(z_h)  # Tensor (batch size, num kernels)
            mu_g5 = self.z_mu_g5(z_h)  # Tensor (batch size, num kernels)
            mu_g6 = self.z_mu_g6(z_h)  # Tensor (batch size, num kernels)
            mu_g7 = self.z_mu_g7(z_h)  # Tensor (batch size, num kernels)
            mu_g8 = self.z_mu_g8(z_h)  # Tensor (batch size, num kernels)
            mu_g9 = self.z_mu_g9(z_h)  # Tensor (batch size, num kernels)
            mu_g10 = self.z_mu_g10(z_h)  # Tensor (batch size, num kernels)

        # Rewrite mean g coefficients in (num kernels, batch size, num coefficients) form
        g_coeffs = torch.full((self.num_kernels, x.shape[0],
                               self.structure.num_tensor_basis), torch.nan)
        for k in range(self.num_kernels):
            g_coeffs[k, :, 0] = mu_g1[:, k]
            g_coeffs[k, :, 1] = mu_g2[:, k]
            g_coeffs[k, :, 2] = mu_g3[:, k]

            if self.structure.num_tensor_basis == 10:
                g_coeffs[k, :, 3] = mu_g4[:, k]
                g_coeffs[k, :, 4] = mu_g5[:, k]
                g_coeffs[k, :, 5] = mu_g6[:, k]
                g_coeffs[k, :, 6] = mu_g7[:, k]
                g_coeffs[k, :, 7] = mu_g8[:, k]
                g_coeffs[k, :, 8] = mu_g9[:, k]
                g_coeffs[k, :, 9] = mu_g10[:, k]

        if self.incl_t0_gen is True:
            mu_g01 = self.z_mu_g01(z_h)  # Tensor (batch size, num kernels)
            mu_g02 = self.z_mu_g02(z_h)  # Tensor (batch size, num kernels)
            mu_g03 = self.z_mu_g03(z_h)  # Tensor (batch size, num kernels)

            t01 = torch.tensor([-1 / 3, 0, 0, 0, 1 / 6, 0, 0, 0, 1 / 6])
            t02 = torch.tensor([1 / 6, 0, 0, 0, -1 / 3, 0, 0, 0, 1 / 6])
            t03 = torch.tensor([1 / 6, 0, 0, 0, 1 / 6, 0, 0, 0, -1 / 3])

        # Calculate mean anisotropy mu_bij
        mu_bij_batch = torch.full((self.num_kernels, x.shape[0], 9), torch.nan)

        for k in range(self.num_kernels):
            for data_point in range(x.shape[0]):
                mu_bij = torch.matmul(g_coeffs[k, data_point, :], tb[data_point].float())

                if self.incl_t0_gen is True:
                    mu_bij = mu_bij + (mu_g01[data_point, k]*t01) + \
                             (mu_g02[data_point, k]*t02) + (mu_g03[data_point, k]*t03)

                # mu_bij_batch[k, data_point, :] = mu_bij
                mu_bij_batch[k, data_point, :] = torch.tanh(mu_bij)

        # Multiply std by number of terms in calculating bij, each with std = sigma
        sigma = self.structure.num_tensor_basis*sigma

        return pi, mu_bij_batch, sigma


class TBMixTVT:
    def __init__(self, optimizer, num_kernels, init_lr, lr_scheduler, lr_scheduler_params,
                 min_epochs, max_epochs, interval, avg_interval, print_freq, log, model):  #
        self.optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=init_lr)
        self.num_kernels = num_kernels
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.interval = interval
        self.avg_interval = avg_interval
        self.print_freq = print_freq
        self.log = log

        # Initialise learning rate scheduler
        lr_scheduler_params = lr_scheduler_params.replace(", ", "=")
        lr_scheduler_params = lr_scheduler_params.split("=")
        param_keys, param_val = [], []
        for i, string in enumerate(lr_scheduler_params):
            if (i+2) % 2 == 0:
                param_keys.append(string)
            else:
                try:
                    param_val.append(float(string))
                except:
                    param_val.append(string)

        lr_scheduler_dict = dict(zip(param_keys, param_val))
        self.lr_scheduler = getattr(torch.optim.lr_scheduler,
                                    lr_scheduler)(self.optimizer, **lr_scheduler_dict)

    def fit(self, train_loader, valid_loader, model):
        # Initialise training
        epoch_count = 0
        train_loss_list, valid_loss_list, max_valid_pi_list = [], [], []
        continue_train = True

        while continue_train:
            # Train model
            epoch_count += 1
            model.train()
            avg_train_loss = self.train_one_epoch(train_loader, self.optimizer, model,
                                                  epoch_count)  # ✓
            train_loss_list.append(avg_train_loss)

            # Print average training MNLL loss per batch
            if epoch_count % self.print_freq == 0:
                print(f"Epoch = {epoch_count}, Average training MNLL loss per "
                      f"batch = {avg_train_loss}")

            # Evaluate model on validation data for early stopping
            if epoch_count % self.interval == 0:
                avg_valid_loss, max_pi = self.perform_valid(valid_loader, model,
                                                            epoch_count)  # ✓
                valid_loss_list.append(avg_valid_loss)
                max_valid_pi_list.append(max_pi)
                continue_train = self.check_conv(valid_loss_list, self.avg_interval,
                                                 epoch_count)  # ✓

                # Print average validation MNLL loss per batch
                print(f"Epoch = {epoch_count}, Average validation MNLL loss per batch = "
                      f"{avg_valid_loss}")
                print("------------------------------------------------------------")

                if continue_train is False:
                    break

            # The check_conv method ensures number of epochs is > min epochs
            # The if statement below ensures number of epochs is <= max epochs
            if epoch_count == self.max_epochs:
                break

            # Update learning rate
            self.lr_scheduler.step()

        # Print final avg train and valid MNLL loss per batch
        self.post_train_print(self.log, epoch_count, train_loss_list[-1],
                              valid_loss_list[-1])  # ✓

        # Plot maximum validation pi and loss curves
        tbnn.plot.plot_scalar(epoch_count, max_valid_pi_list, self.interval, "Max pi")
        tbnn.plot.plot_loss_curves(epoch_count, train_loss_list, valid_loss_list,
                                   self.interval)

        return epoch_count, train_loss_list, valid_loss_list

    def train_one_epoch(self, train_loader, optimizer, model, epoch_count):  #
        # Run one training epoch on the model
        running_train_loss = 0
        batch_idx = 0
        skip_batch_idx = [0]*len(train_loader)

        for x, tb, y in train_loader:
            # Forward propagation
            batch_idx += 1
            pi, mu_bij, sigma = model(x, tb)  # ✓

            # Skip batch if model outputs give nan
            if torch.any(torch.isnan(pi)).item() is True or \
                    torch.any(torch.isnan(mu_bij)).item() is True or \
                    torch.any(torch.isnan(sigma)).item() is True:
                skip_batch_idx[batch_idx-1] = batch_idx
                continue

            # Calculate mean negative log likelihood loss for current batch
            loss = self.mnll_loss(pi, mu_bij, sigma, y, model)  # ✓
            if torch.isnan(loss).item() is True:
                raise Exception("NaN loss detected")
            if torch.isinf(loss).item() is True:
                skip_batch_idx[batch_idx-1] = batch_idx
                continue

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5e-5)
            optimizer.step()

            # Sum MNLL loss
            running_train_loss += loss.item()

        # Print skipped batches
        skip_batch_idx = list(filter(None, skip_batch_idx))
        if skip_batch_idx:
            print(f"Epoch {epoch_count} skipped batches {skip_batch_idx} due to low "
                  f"sigma")
            print(f"Epoch {epoch_count}: {len(skip_batch_idx)} / {len(train_loader)} "
                  f"batches skipped")

        # Return MNLL loss per batch
        return running_train_loss / (len(train_loader) - len(skip_batch_idx))

    def mnll_loss(self, pi, mu_bij_batch, sigma, y, model, log=True, reg=None):
        # Make y, sigma, and pi have same dimensions as mu_bij_batch
        y = y.unsqueeze(0).repeat(self.num_kernels, 1, 1)
        sigma = torch.transpose(sigma, 0, 1).unsqueeze(2).repeat(1, 1, 9)
        pi = torch.transpose(pi, 0, 1).unsqueeze(2).repeat(1, 1, 9)
        c = 9  # Number of output variables

        # Calculate distribution difference
        diff = (y - mu_bij_batch) * torch.reciprocal(sigma)
        diff = 0.5 * torch.square(diff)

        if log is True and self.num_kernels == 1:
            # Calculate loss in the log domain
            part_1 = (1/2)*math.log(2*np.pi)
            part_2 = (1/y.shape[1])*torch.sum(torch.log(sigma[0, :, 0]))
            part_3 = (1/(y.shape[1]*9))*torch.sum(diff)
            nll_loss = part_1 + part_2 + part_3
        else:
            # Calculate loss in standard manner
            frac = 1.0 / (2.0 * np.pi)**(c/2)
            gauss_diff = frac * torch.reciprocal(torch.pow(sigma, c)) * torch.exp(-diff)
            assert torch.all(gauss_diff >= 0).tolist() is True

            # Calculate negative log-likelihood loss
            nll_loss = torch.sum(pi * gauss_diff, dim=0)  # sum over kernels
            nll_loss = -torch.log(nll_loss)
            nll_loss = torch.mean(nll_loss)

        # Calculate L2 regularization and return combined loss, else return nll loss
        if reg == "l2":
            reg_lambda = 10
            param_sum = 0
            for param in model.parameters():
                param_sum += param.pow(2.0).sum()
            l2_reg = reg_lambda * param_sum
            return nll_loss + l2_reg
        else:
            return nll_loss

    def perform_valid(self, valid_loader, model, epoch_count):
        # Evaluate model on validation data
        running_valid_loss = 0
        batch_idx = 0
        skip_batch_idx = [0]*len(valid_loader)
        max_pi = 0

        with torch.no_grad():
            model.eval()
            for x, tb, y in valid_loader:
                # Forward propagation
                batch_idx += 1
                pi, mu_bij, sigma = model(x, tb)  # ✓

                # Calculate MNLL loss for current batch
                loss = self.mnll_loss(pi, mu_bij, sigma, y, model)  # ✓
                if torch.isnan(loss).item() is True:  # Check for loss = nan
                    raise Exception("NaN loss detected")
                if torch.isinf(loss).item() is True:  # Check for loss = inf
                    skip_batch_idx[batch_idx-1] = batch_idx
                    continue

                # Sum MNLL loss
                running_valid_loss += loss.item()

                # Update max pi
                if torch.max(pi).item() > max_pi:
                    max_pi = torch.max(pi).item()

        # Print number of batches skipped due to low sigma
        print("------------------------------------------------------------")
        skip_batch_idx = list(filter(None, skip_batch_idx))
        if skip_batch_idx:
            print(f"Epoch {epoch_count}: {len(skip_batch_idx)} / {len(valid_loader)} "
                  f"batches skipped")

        # Return average MNLL loss per batch
        return running_valid_loss / (len(valid_loader) - len(skip_batch_idx)), max_pi

    def check_conv(self, valid_loss_list, avg_interval, epoch_count):  #
        # Activate early stopping if validation error starts increasing
        if epoch_count >= self.min_epochs:
            return np.mean(valid_loss_list[-avg_interval:]) < \
                   np.mean(valid_loss_list[-2 * avg_interval:-avg_interval])
        else:
            return True

    @staticmethod
    def post_train_print(log, epoch_count, final_avg_train_loss, final_avg_valid_loss):  #
        # Print average MNLL losses per batch in console
        print("============================================================")
        print(f"Total number of epochs = {epoch_count}, "
              f"Final average training MNLL loss per batch = {final_avg_train_loss},"
              f"Final average validation MNLL loss per batch = {final_avg_valid_loss}")

        # Print average MNLL losses per batch in log file
        with open(log, "a") as write_file:
            print("Total number of epochs = ", epoch_count, file=write_file)
            print("Final average training MNLL loss per batch = ",
                  final_avg_train_loss, file=write_file)
            print("Final average validation MNLL loss per batch = ",
                  final_avg_valid_loss, file=write_file)

    def test_preprocessing(self, x_test, tb_test, y_test):
        # Convert data from np.array to torch.tensor
        x_test = torch.from_numpy(x_test)
        tb_test = torch.from_numpy(tb_test)
        y_test = torch.from_numpy(y_test)

        # Predict on test dataset
        pi_all = torch.full((len(x_test), self.num_kernels), torch.nan)
        mu_bij_all = torch.full((self.num_kernels, len(x_test), 9), torch.nan)
        sigma_all = torch.full((len(x_test), self.num_kernels), torch.nan)
        mu_bij_pred = torch.full((len(x_test), 9), torch.nan)

        return x_test, tb_test, y_test, pi_all, mu_bij_all, sigma_all, mu_bij_pred

    @staticmethod
    def find_test_neighbours(test_list, coords_test):
        # Find neighbouring data points in the flow domain using a Voronoi diagram
        assert len(test_list) == 1
        vor = Voronoi(coords_test)
        pairs = vor.ridge_points
        neigh_dict = {k: [] for k in range(len(coords_test))}

        # Fill neighbours dictionary {coord_idx:neighbour coord_idx}
        for p in pairs:
            neigh_dict[p[0]].append(p[1])
            neigh_dict[p[1]].append(p[0])

        return neigh_dict

    def perform_most_prob_test(self, x_test, tb_test, y_test, model, pi_all, mu_bij_all,
                               sigma_all, mu_bij_pred, enforce_realiz, num_realiz_its,
                               log):
        with torch.no_grad():
            model.eval()
            running_test_loss = 0

            # Obtain all and most probable kernel predictions
            for i in range(len(x_test)):
                pi, mu_bij, sigma = model(torch.unsqueeze(x_test[i, :], 0),
                                          torch.unsqueeze(tb_test[i, :, :], 0))  # ✓
                pi_all[i, :] = pi
                for k in range(self.num_kernels):
                    mu_bij_all[k, i, :] = mu_bij[k, :, :]
                sigma_all[i, :] = sigma
                loss = self.mnll_loss(pi, mu_bij, sigma, y_test[i, :], model) # ✓
                running_test_loss += loss.item()

                k = torch.argmax(pi).item()
                mu_bij_pred[i, :] = mu_bij[k, :, :]

        avg_mnll_loss = running_test_loss / len(x_test)
        print(f"Max value of pi = {torch.max(pi_all).item()}")

        # Convert mu_bij_pred to np array and enforce realizability
        mu_bij_pred = mu_bij_pred.numpy()
        y_test = y_test.numpy()

        # if enforce_realiz:
        #     for i in range(num_realiz_its):
        #         mu_bij_pred = tbnn.calc.PopeDataProcessor.make_realizable(mu_bij_pred)

        # Calculate, print and write testing RMSE
        test_rmse = self.calc_test_rmse(mu_bij_pred, y_test, log)  # ✓

        # Convert tensors to np arrays to view them
        pi_all = pi_all.numpy()
        mu_bij_all = mu_bij_all.numpy()
        sigma_all = sigma_all.numpy()

        return mu_bij_pred, avg_mnll_loss, test_rmse, pi_all, mu_bij_all, sigma_all

    def perform_anchors_test(self, x_test, tb_test, y_test, model, pi_all, mu_bij_all,
                             sigma_all, mu_bij_pred, neigh_dict, enforce_realiz,
                             num_realiz_its, log):  #
        with torch.no_grad():
            model.eval()

            # First testing round: Find points that have high pi
            running_test_loss = 0
            tf_list = [False] * len(x_test)
            for i in range(len(x_test)):
                pi, mu_bij, sigma = model(torch.unsqueeze(x_test[i, :], 0),
                                          torch.unsqueeze(tb_test[i, :, :], 0))  # ✓
                pi_all[i, :] = pi
                for k in range(self.num_kernels):
                    mu_bij_all[k, i, :] = mu_bij[k, :, :]
                sigma_all[i, :] = sigma
                loss = self.mnll_loss(pi, mu_bij, sigma, y_test[i, :], model) # ✓
                running_test_loss += loss.item()

                for k, p in enumerate(torch.squeeze(pi)):
                    if p.item() > 0.7:
                        assert torch.all(torch.isnan(mu_bij_pred[i, :])).item() is True
                        mu_bij_pred[i, :] = mu_bij[k, :, :]
                        tf_list[i] = True

        avg_nll_loss = running_test_loss / (len(x_test) * 9)
        print(f"Max value of pi = {torch.max(pi_all).item()}")

        # Second testing round: Find points near those from the previous testing round
        # and choose the closest bij prediction to the points that passed the previous
        # testing round

        # Find differences between current tf_list and previous tf_list
        anchor_loops = 0
        prev_tf_list = [False] * len(x_test)
        any_false = True

        while any_false:
            tf_diff = [tf2 - tf1 for (tf1, tf2) in zip(prev_tf_list, tf_list)]
            if any(tf_diff) is False:
                raise Exception(f"{sum(tf_list)} points filled out of {len(tf_list)} total"
                                f" points with no neighbouring points to choose from")

            anchor_loops += 1
            print(f"Anchor loop {anchor_loops} starting with {sum(tf_list)} filled")
            prev_tf_list = list(tf_list)
            for i, tf in enumerate(tf_diff):
                if tf == 1:
                    for n in neigh_dict[i]:
                        if torch.all(torch.isnan(mu_bij_pred[n, :])).item() is False:
                            continue
                        else:
                            assert torch.all(torch.isnan(mu_bij_pred[n, :])).item() is True
                            mu_bij_diff = mu_bij_pred[i, :].repeat(self.num_kernels, 1) \
                                - mu_bij_all[:, n, :]
                            diff_sum = torch.sum(torch.abs(mu_bij_diff), dim=1)
                            k = torch.argmin(diff_sum).item()
                            mu_bij_pred[n, :] = mu_bij_all[k, n, :]
                            tf_list[n] = True

            if all(tf_list) is True:
                break

        # Convert mu_bij_pred to np array and enforce realizability
        mu_bij_pred = mu_bij_pred.numpy()
        y_test = y_test.numpy()

        # if enforce_realiz:
        #     for i in range(num_realiz_its):
        #         mu_bij_pred = tbnn.calc.PopeDataProcessor.make_realizable(mu_bij_pred)

        # Calculate, print and write testing RMSE
        test_rmse = self.calc_test_rmse(mu_bij_pred, y_test, log)  # ✓

        # Convert tensors to np arrays to view them
        pi_all = pi_all.numpy()
        mu_bij_all = mu_bij_all.numpy()
        sigma_all = sigma_all.numpy()

        return mu_bij_pred, avg_nll_loss, test_rmse, pi_all, mu_bij_all, sigma_all

    @staticmethod
    def calc_test_rmse(y_pred, y_test, log):  #
        # Calculate testing RMSE, then print to console and write to file
        assert y_pred.shape[0] == y_test.shape[0], \
            "Number of rows in y_pred and y_test are different"
        assert y_pred.shape[1] == 9, \
            "Number of columns in y_pred does not equal the number of bij components"

        test_rmse = np.sqrt(np.mean(np.square(y_test - y_pred)))
        print("Testing RMSE per data point per bij comp = ", test_rmse)
        with open(log, "a") as write_file:
            print("Testing RMSE per data point per bij comp = ", test_rmse,
                  file=write_file)

        return test_rmse
