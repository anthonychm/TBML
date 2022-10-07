import numpy as np
import time
import torch
import torch.nn as nn
import torch.utils.data as torchdata
import pandas as pd
from calculator import PopeDataProcessor


class DataLoader:
    def __init__(self, x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test, tb_test, y_test, batch_size):

        # Convert data from np.array into torch.tensor
        self.x_train = torch.from_numpy(x_train)
        self.tb_train = torch.from_numpy(tb_train)
        self.y_train = torch.from_numpy(y_train)
        self.x_valid = torch.from_numpy(x_valid)
        self.tb_valid = torch.from_numpy(tb_valid)
        self.y_valid = torch.from_numpy(y_valid)
        self.x_test = torch.from_numpy(x_test)
        self.tb_test = torch.from_numpy(tb_test)
        self.y_test = torch.from_numpy(y_test)

        # Run TensorDataset and DataLoader for batch allocation
        train_dataset = torchdata.TensorDataset(self.x_train, self.tb_train, self.y_train)
        valid_dataset = torchdata.TensorDataset(self.x_valid, self.tb_valid, self.y_valid)
        test_dataset = torchdata.TensorDataset(self.x_test, self.tb_test, self.y_test)
        # Order of data points already shuffled during train-validation-test splits, so we can set shuffle=False here
        self.train_loader = torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.valid_loader = torchdata.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader = torchdata.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        def print_batch(loader, dataset_type):
            print("First batch of ", dataset_type, "dataset:")
            data_iter = iter(loader)
            print(next(data_iter))

        # Print first batch to verify data loader database size and determinism
        print_batch(self.train_loader, "training")
        print_batch(self.valid_loader, "validation")
        print_batch(self.test_loader, "testing")
        print("Data loaders created")

    @staticmethod
    def check_data_loaders(data_loader, x, tb, y, num_inputs, num_tensor_basis):
        # Check that the invariants inputs, tensor basis inputs, and bij outputs all have the same number of data points
        assert x.shape[0] == y.shape[0], "Mismatched shapes between inputs and outputs"
        assert x.shape[0] == tb.shape[0], "Mismatched shapes between inputs and tensors"

        # Check data loader dataset shapes
        assert data_loader.dataset.tensors[0].size(dim=0) == x.shape[0], "Error in data loader dataset shape"
        assert data_loader.dataset.tensors[0].size(dim=1) == num_inputs, "Error in data loader dataset shape"
        assert data_loader.dataset.tensors[1].size(dim=0) == tb.shape[0], "Error in data loader dataset shape"
        assert data_loader.dataset.tensors[1].size(dim=1) == num_tensor_basis, "Error in data loader dataset shape"
        assert data_loader.dataset.tensors[1].size(dim=2) == 9, "Error in data loader dataset shape"
        assert data_loader.dataset.tensors[2].size(dim=0) == y.shape[0], "Error in data loader dataset shape"
        assert data_loader.dataset.tensors[2].size(dim=1) == 9, "Error in data loader dataset shape"


class NetworkStructure:
    """
    A class to define the layer structure for the neural network
    """
    def __init__(self, num_hid_layers, num_hid_nodes, af, af_params, num_inputs, num_tensor_basis):
        self.num_hid_layers = num_hid_layers  # Number of hidden layers
        self.num_hid_nodes = num_hid_nodes  # Number of nodes per hidden layer
        self.num_inputs = num_inputs  # Number of scalar invariants
        self.num_tensor_basis = num_tensor_basis  # Number of tensors in the tensor basis
        self.af = af  # Activation functions

        # Create activation function arguments pandas dataframe
        af_df = pd.Series(af_params)
        self.af_df = af_df.str.split(pat=", ")

    def check_structure(self):
        # Check that the number of hidden nodes, activation functions and their parameters are consistent with the
        # number of hidden layers
        assert len(self.num_hid_nodes) == self.num_hid_layers, \
            "Mismatch between the length of num_hid_nodes and value of num_hid_layers"
        assert len(self.af) == self.num_hid_layers, "Mismatch between the length of af and value of num_hid_layers"
        assert len(self.af_df) == self.num_hid_layers, "Mismatch between the length of af_params and value of " \
                                                         "num_hid_layers"


class Tbnn(nn.Module):
    def __init__(self, device, seed, structure=None, weight_init="dummy", weight_init_params="dummy"):
        super(Tbnn, self).__init__()
        if structure is None:
            print("Network structure not defined")
            raise Exception
        self.structure = structure
        self.seed = seed
        self.weight_init = weight_init
        weight_init_params = weight_init_params.replace(", ", "=")
        self.weight_init_params = weight_init_params.split("=")

        def retrieve_af_params(af_df, layer):
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
        layer = 0
        self.net.add_module("layer1", nn.Linear(self.structure.num_inputs, self.structure.num_hid_nodes[layer]))
        af_param_dict = retrieve_af_params(self.structure.af_df, layer)
        self.net.add_module("af1", getattr(nn, self.structure.af[layer])(**af_param_dict))
        for layer in range(1, self.structure.num_hid_layers):
            self.net.add_module("layer"+str(layer+1), nn.Linear(self.structure.num_hid_nodes[layer-1],
                                                                self.structure.num_hid_nodes[layer]))
            af_param_dict = retrieve_af_params(self.structure.af_df, layer)
            self.net.add_module("af" + str(layer+1), getattr(nn, self.structure.af[layer])(**af_param_dict))
        self.net.add_module("coeffs_layer", nn.Linear(self.structure.num_hid_nodes[-1], self.structure.num_tensor_basis))
        print("Building of NN complete")

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
        self.net = self.net.to(device)
        print("Weights and biases initialized")

    # Forward propagation
    def forward(self, x, tb, device):
        coeffs = self.net(x)
        for data_point in range(0, len(coeffs)):
            for bij_comp in range(0, 9):
                tensors = torch.index_select(tb[data_point], 1, torch.tensor([bij_comp]).to(device))
                tensors = torch.squeeze(tensors, dim=1)
                bij_tmp = torch.dot(coeffs[data_point], tensors)
                bij_tmp = torch.unsqueeze(bij_tmp, dim=0)
                if "bij_row" in locals():
                    bij_row = torch.cat((bij_row, bij_tmp), dim=0)
                else:
                    bij_row = bij_tmp
            bij_row = torch.unsqueeze(bij_row, dim=0)

            if "bij_batch_pred" in locals():
                bij_batch_pred = torch.cat((bij_batch_pred, bij_row), dim=0)
            else:
                bij_batch_pred = bij_row
            del bij_row

        return bij_batch_pred


class TbnnTVT:
    def __init__(self, loss, optimizer, init_lr, lr_scheduler, lr_scheduler_params, min_epochs, max_epochs, interval,
                 avg_interval, print_freq, log, model):
        self.criterion = getattr(torch.nn, loss)()
        self.optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=init_lr)
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
        self.lr_scheduler = getattr(torch.optim.lr_scheduler, lr_scheduler)(self.optimizer, **lr_scheduler_dict)

    def fit(self, device, train_loader, valid_loader, model):
        epoch_count = 0
        valid_loss_list = []
        continue_train = True

        # Training loop
        print("Epoch    Average training RMSE per data point    Average validation RMSE per data point")
        while continue_train is True:
            epoch_count += 1
            model.train()
            epoch_count, avg_train_loss = self.train_one_epoch(train_loader, self.criterion, self.optimizer, device,
                                                               model, epoch_count)

            # Predict on validation data for early stopping
            if epoch_count % self.interval == 0:
                avg_valid_loss = self.perform_valid(valid_loader, device, model, self.criterion)
                valid_loss_list.append(avg_valid_loss)
                continue_train = self.check_conv(valid_loss_list, self.min_epochs, self.avg_interval, epoch_count)

            # Ensure number of epochs is > min epochs and < max epochs
            # The two if statements below override the one above
            if epoch_count < self.min_epochs:
                continue_train = True

            if epoch_count > self.max_epochs:
                continue_train = False

            # Print average training and validation RMSEs per data point in console
            if epoch_count % self.print_freq == 0:
                print(epoch_count, np.sqrt(avg_train_loss), np.sqrt(avg_valid_loss))

            # Update learning rate
            self.lr_scheduler.step()

        # Output final training and validation RMSEs per data point
        final_train_rmse = np.sqrt(avg_train_loss)
        final_valid_rmse = np.sqrt(avg_valid_loss)
        self.post_train_print(self.log, epoch_count, final_train_rmse, final_valid_rmse)

        return epoch_count, final_train_rmse, final_valid_rmse

    def train_one_epoch(self, train_loader, criterion, optimizer, device, model, epoch_count):
        running_train_loss = 0

        for i, (x, tb, y) in enumerate(train_loader):
            x, tb, y = self.to_device(x, tb, y, device)

            # Forward propagation
            bij_batch_pred = model(x, tb, device)
            loss = criterion(bij_batch_pred, y)

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record loss
            running_train_loss += loss.item()

            # if i == 1:
            #     print("Batch 1 invariants, x = ", x)
            #     print("Batch 1 tensor basis, tb = ", tb)
            #     print("Batch 1 true bij = ", y)
            #     print("Batch 1 bij predictions = ", bij_batch_pred)

        avg_train_loss = running_train_loss / len(train_loader)  # average loss per data point

        return epoch_count, avg_train_loss

    def perform_valid(self, valid_loader, device, model, criterion):
        # Predict on validation data
        running_valid_loss = 0
        with torch.no_grad():
            model.eval()
            for i, (x, tb, y) in enumerate(valid_loader):
                x, tb, y = self.to_device(x, tb, y, device)
                valid_outputs = model(x, tb, device)
                valid_loss = criterion(valid_outputs, y)
                running_valid_loss += valid_loss.item()

        avg_valid_loss = running_valid_loss / len(valid_loader)  # average loss per data point

        return avg_valid_loss

    def perform_test(self, device, enforce_realiz, num_realiz_its, log, test_loader, model):
        # Predict on test data
        with torch.no_grad():
            model.eval()
            for x, tb, y in test_loader:
                x, tb, y = self.to_device(x, tb, y, device)
                y_pred_tmp = model(x, tb, device)
                if "y_pred" in locals():
                    y_pred = torch.vstack((y_pred, y_pred_tmp))
                else:
                    y_pred = y_pred_tmp

        # Convert y_pred to np array and enforce realizability
        y_pred = y_pred.detach().cpu().numpy()
        if enforce_realiz:
            for i in range(num_realiz_its):
                y_pred = PopeDataProcessor.make_realizable(y_pred)

        # Calculate, print and write testing RMSE
        test_rmse = self.test_rmse_ops(y_pred, test_loader, log)

        return y_pred, test_rmse

    @staticmethod
    def to_device(x, tb, y, device):
        x = x.to(device)
        tb = tb.to(device)
        y = y.to(device)
        return x, tb, y

    @staticmethod
    def check_conv(valid_loss_list, min_epochs, avg_interval, epoch_count):
        if epoch_count > min_epochs:
            # Activate early stopping if validation error starts increasing
            continue_train = np.mean(valid_loss_list[-avg_interval:]) < \
                             np.mean(valid_loss_list[-2*avg_interval:-avg_interval])

            return continue_train

    @staticmethod
    def post_train_print(log, epoch_count, final_train_rmse, final_valid_rmse):
        # Print training and validation RMSEs to console and write to file
        print("Total number of epochs = ", epoch_count)
        print("Final average training RMSE per data point = ", final_train_rmse)
        print("Final average validation RMSE per data point = ", final_valid_rmse)

        with open(log, "a") as write_file:
            print("Total number of epochs = ", epoch_count, file=write_file)
            print("Final average training RMSE per data point = ", final_train_rmse, file=write_file)
            print("Final average validation RMSE per data point = ", final_valid_rmse, file=write_file)

    @staticmethod
    def test_rmse_ops(y_pred, test_loader, log):
        # Calculate testing RMSE, then print to console and write to file
        assert y_pred.shape[0] == len(test_loader.dataset), "Number of rows in y_pred and test loader are different"
        assert y_pred.shape[1] == 9, "Number of columns in y_pred does not equal the number of bij components"

        y = test_loader.dataset.tensors[2].detach().cpu().numpy()
        test_rmse = np.sqrt(np.mean(np.square(y - y_pred))) # to do: Check this
        print("Testing RMSE per datapoint = ", test_rmse)

        with open(log, "a") as write_file:
            print("Testing RMSE per datapoint = ", test_rmse, file=write_file)

        return test_rmse
