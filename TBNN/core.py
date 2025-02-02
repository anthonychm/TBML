import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torchdata
import pandas as pd
from calculator import PopeDataProcessor


class DataLoader:
    def __init__(self, x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test,
                 tb_test, y_test, batch_size):  # ✓

        # Convert data from np.array into torch.tensor ✓
        self.x_train = torch.from_numpy(x_train)
        self.tb_train = torch.from_numpy(tb_train)
        self.y_train = torch.from_numpy(y_train)
        self.x_valid = torch.from_numpy(x_valid)
        self.tb_valid = torch.from_numpy(tb_valid)
        self.y_valid = torch.from_numpy(y_valid)
        self.x_test = torch.from_numpy(x_test)
        self.tb_test = torch.from_numpy(tb_test)
        self.y_test = torch.from_numpy(y_test)

        # Run TensorDataset and DataLoader for batch allocation ✓
        train_dataset = torchdata.TensorDataset(self.x_train, self.tb_train, self.y_train)
        valid_dataset = torchdata.TensorDataset(self.x_valid, self.tb_valid, self.y_valid)
        test_dataset = torchdata.TensorDataset(self.x_test, self.tb_test, self.y_test)
        # Order of data points already shuffled during train-validation-test splits, so
        # we can set shuffle=False here ✓
        self.train_loader = torchdata.DataLoader(train_dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        self.valid_loader = torchdata.DataLoader(valid_dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        self.test_loader = torchdata.DataLoader(test_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=0)
        # Note: While validation and testing do not require batches, they are still used
        # to reduce memory requirements

        def print_batch(loader, dataset_type):  # ✓
            print("First batch of ", dataset_type, "dataset:")
            data_iter = iter(loader)
            print(next(data_iter))

        # Print first batch to verify data loader database size and determinism ✓
        print_batch(self.train_loader, "training")  # ✓
        print_batch(self.valid_loader, "validation")  # ✓
        print_batch(self.test_loader, "testing")  # ✓
        print("Data loaders created")

    @staticmethod
    def check_data_loaders(data_loader, x, tb, y, num_inputs, num_tensor_basis):  # ✓
        # Check that scalar inputs, tensor basis inputs, and bij outputs all have the
        # same number of data points
        assert x.shape[0] == y.shape[0], "Mismatched shapes between inputs and outputs"
        assert x.shape[0] == tb.shape[0], "Mismatched shapes between inputs and tensors"

        # Check data loader dataset shapes
        shape_error = "Error in data loader dataset shape"
        assert data_loader.dataset.tensors[0].size(dim=0) == x.shape[0], shape_error
        assert data_loader.dataset.tensors[0].size(dim=1) == num_inputs, shape_error
        assert data_loader.dataset.tensors[1].size(dim=0) == tb.shape[0], shape_error
        assert data_loader.dataset.tensors[1].size(dim=1) == num_tensor_basis, shape_error
        assert data_loader.dataset.tensors[1].size(dim=2) == 9, shape_error
        assert data_loader.dataset.tensors[2].size(dim=0) == y.shape[0], shape_error
        assert data_loader.dataset.tensors[2].size(dim=1) == 9, shape_error


class NetworkStructure:
    """
    A class to define the layer structure for the neural network
    """
    def __init__(self, num_hid_layers, num_hid_nodes, af, af_params, num_inputs,
                 num_tensor_basis):  # ✓
        self.num_hid_layers = num_hid_layers  # Number of hidden layers
        self.num_hid_nodes = num_hid_nodes  # Number of nodes per hidden layer
        self.num_inputs = num_inputs  # Number of scalar inputs
        self.num_tensor_basis = num_tensor_basis  # Number of tensors in the tensor basis
        self.af = af  # Activation functions

        # Create activation function arguments pandas dataframe
        if af_params is not None:
            af_df = pd.Series(af_params)
            self.af_df = af_df.str.split(pat=", ")
        else:
            self.af_df = None

    def check_structure(self):  # ✓
        # Check that the number of hidden nodes, activation functions and their
        # parameters are consistent with the number of hidden layers
        assert len(self.num_hid_nodes) == self.num_hid_layers, \
            "Mismatch between the length of num_hid_nodes and value of num_hid_layers"
        assert len(self.af) == self.num_hid_layers, \
            "Mismatch between the length of af and value of num_hid_layers"
        if self.af_df is not None:
            assert len(self.af_df) == self.num_hid_layers, \
                "Mismatch between the length of af_params and value of num_hid_layers"


class Tbnn(nn.Module):
    def __init__(self, device, seed, structure=None, weight_init="dummy",
                 weight_init_params="dummy", incl_t0_gen=False):  # ✓
        super(Tbnn, self).__init__()
        if structure is None:
            print("Network structure not defined")
            raise Exception
        self.structure = structure
        self.seed = seed
        self.weight_init = weight_init
        weight_init_params = weight_init_params.replace(", ", "=")
        self.weight_init_params = weight_init_params.split("=")
        self.incl_t0_gen = incl_t0_gen

        def retrieve_af_params(af_df, layer):  # ✓
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

        # Create PyTorch neural network ✓
        self.net = nn.Sequential()
        layer = 0
        self.net.add_module("layer1", nn.Linear(self.structure.num_inputs,
                                                self.structure.num_hid_nodes[layer]))
        if self.structure.af_df is None:
            self.net.add_module("af1", getattr(nn, self.structure.af[layer])())
        else:
            af_param_dict = retrieve_af_params(self.structure.af_df, layer)  # ✓
            self.net.add_module("af1",
                                getattr(nn, self.structure.af[layer])(**af_param_dict))

        for layer in range(1, self.structure.num_hid_layers):
            self.net.add_module("layer"+str(layer+1),
                                nn.Linear(self.structure.num_hid_nodes[layer-1],
                                          self.structure.num_hid_nodes[layer]))
            if self.structure.af_df is None:
                self.net.add_module("af" + str(layer+1),
                                    getattr(nn, self.structure.af[layer])())
            else:
                af_param_dict = retrieve_af_params(self.structure.af_df, layer)  # ✓
                self.net.add_module("af" + str(layer+1),
                                    getattr(nn, self.structure.af[layer])(**af_param_dict))
        self.z_coeffs = nn.Linear(self.structure.num_hid_nodes[-1],
                                  self.structure.num_tensor_basis)

        # Include hidden layer outputs for t0_gen if required
        if self.incl_t0_gen is True:
            self.z_g01 = nn.Linear(self.structure.num_hid_nodes[-1], 1)
            self.z_g02 = nn.Linear(self.structure.num_hid_nodes[-1], 1)
            self.z_g03 = nn.Linear(self.structure.num_hid_nodes[-1], 1)

        print("Building of neural network complete")

        # Create initialization arguments dictionary and initialize weights and biases ✓
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
        self.z_coeffs.apply(init_w_and_b)
        self.z_coeffs = self.z_coeffs.to(device)

        if self.incl_t0_gen is True:
            self.z_g01.apply(init_w_and_b)
            self.z_g01 = self.z_g01.to(device)
            self.z_g02.apply(init_w_and_b)
            self.z_g02 = self.z_g02.to(device)
            self.z_g03.apply(init_w_and_b)
            self.z_g03 = self.z_g03.to(device)

        print("Weights and biases initialized")

    # Forward propagation
    def forward(self, x, tb, device):
        # Predict coefficients in forward pass
        z_h = self.net(x)
        coeffs = self.z_coeffs(z_h)

        def pred_t0_gen():
            # Predict generalised T0 and use it to initialise bij_batch_pred tensor
            g01 = self.z_g01(z_h)
            g02 = self.z_g02(z_h)
            g03 = self.z_g03(z_h)

            t01 = torch.tensor([[-1 / 3, 0, 0, 0, 1 / 6, 0, 0, 0, 1 / 6]])
            t02 = torch.tensor([[1 / 6, 0, 0, 0, -1 / 3, 0, 0, 0, 1 / 6]])
            t03 = torch.tensor([[1 / 6, 0, 0, 0, 1 / 6, 0, 0, 0, -1 / 3]])

            return torch.mul(g01, t01) + torch.mul(g02, t02) + torch.mul(g03, t03)

        # Initialise bij batch prediction tensor
        if self.incl_t0_gen is True:
            bij_batch_pred = pred_t0_gen().double()
        else:
            bij_batch_pred = torch.zeros(len(coeffs), 9, dtype=torch.double)

        # Populate bij batch prediction tensor
        for point in range(len(coeffs)):
            for bij_comp in range(9):
                tensors = torch.index_select(tb[point], 1,
                                             torch.tensor([bij_comp]).to(device))
                tensors = torch.squeeze(tensors, dim=1)
                bij_batch_pred[point, bij_comp] += torch.dot(coeffs[point], tensors)

        return bij_batch_pred


class TbnnTVT:
    def __init__(self, loss, optimizer, init_lr, lr_scheduler, lr_scheduler_params,
                 min_epochs, max_epochs, interval, avg_interval, print_freq, log,
                 model):  # ✓
        self.criterion = getattr(torch.nn, loss)()
        self.optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=init_lr)
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.interval = interval
        self.avg_interval = avg_interval
        self.print_freq = print_freq
        self.log = log

        # Initialise learning rate scheduler ✓
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

    def fit(self, device, train_loader, valid_loader, model):  # ✓
        epoch_count = 0
        valid_loss_list = []
        continue_train = True

        # Training loop ✓
        while continue_train:
            epoch_count += 1
            model.train()
            avg_train_loss = self.train_one_epoch(train_loader, self.criterion,
                                                  self.optimizer, device, model)  # ✓

            # Print average training rmse per data point per bij comp
            if epoch_count % self.print_freq == 0:
                print(f"Epoch = {epoch_count}, Average training rmse per data point per "
                      f"bij component = {np.sqrt(avg_train_loss)}")

            # Predict on validation data for early stopping ✓
            if epoch_count % self.interval == 0:
                avg_valid_loss = self.perform_valid(valid_loader, device, model,
                                                    self.criterion)  # ✓
                valid_loss_list.append(avg_valid_loss)
                continue_train = self.check_conv(valid_loss_list, self.min_epochs,
                                                 self.avg_interval, epoch_count)  # ✓

                # Print average validation rmse per data point per bij comp
                print("------------------------------------------------------------")
                print(f"Epoch = {epoch_count}, Average validation rmse per data point "
                      f"per bij component = {np.sqrt(avg_valid_loss)}")
                print("------------------------------------------------------------")

                if continue_train is False:
                    break

            # The check_conv method ensures number of epochs is > min epochs
            # The if statement below ensures number of epochs is < max epochs ✓
            if epoch_count == self.max_epochs:
                break

            # Update learning rate
            self.lr_scheduler.step()

        # Output final training and validation RMSEs per data point ✓
        final_train_rmse = np.sqrt(avg_train_loss)
        final_valid_rmse = np.sqrt(avg_valid_loss)
        self.post_train_print(self.log, epoch_count, final_train_rmse,
                              final_valid_rmse)  # ✓

        return epoch_count, final_train_rmse, final_valid_rmse

    def train_one_epoch(self, train_loader, criterion, optimizer, device, model):  # ✓
        running_train_loss = 0

        for i, (x, tb, y) in enumerate(train_loader):
            x, tb, y = self.to_device(x, tb, y, device)  # ✓

            # Forward propagation
            bij_batch_pred = model(x, tb, device)  # ✓
            # Calculate MSE per datapoint per bij comp for entire batch
            loss = criterion(bij_batch_pred, y)

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record loss
            running_train_loss += loss.item()  # Sum of MSE for all batches

        # Calculate average loss per data point per bij comp
        avg_train_loss = running_train_loss / len(train_loader)

        return avg_train_loss

    def perform_valid(self, valid_loader, device, model, criterion):  # ✓
        # Predict on validation data
        running_valid_loss = 0
        with torch.no_grad():
            model.eval()
            for i, (x, tb, y) in enumerate(valid_loader):
                x, tb, y = self.to_device(x, tb, y, device)  # ✓
                valid_outputs = model(x, tb, device)  # ✓
                valid_loss = criterion(valid_outputs, y)
                running_valid_loss += valid_loss.item()

        # Calculate average loss per data point per bij comp
        avg_valid_loss = running_valid_loss / len(valid_loader)

        return avg_valid_loss

    def perform_test(self, device, enforce_realiz, num_realiz_its, log, test_loader,
                     model):  # ✓
        # Predict on test data ✓
        with torch.no_grad():
            model.eval()
            for x, tb, y in test_loader:
                x, tb, y = self.to_device(x, tb, y, device)  # ✓
                y_pred_tmp = model(x, tb, device)  # ✓
                if "y_pred" in locals():
                    y_pred = torch.vstack((y_pred, y_pred_tmp))
                else:
                    y_pred = y_pred_tmp

        # Convert y_pred to np array and enforce realizability ✓
        y_pred = y_pred.detach().cpu().numpy()
        if enforce_realiz:
            for i in range(num_realiz_its):
                y_pred = PopeDataProcessor.make_realizable(y_pred)

        # Calculate, print and write testing RMSE ✓
        test_rmse = self.test_rmse_ops(y_pred, test_loader, log)  # ✓

        return y_pred, test_rmse

    @staticmethod
    def to_device(x, tb, y, device):  # ✓
        x = x.to(device)
        tb = tb.to(device)
        y = y.to(device)
        return x, tb, y

    @staticmethod
    def check_conv(valid_loss_list, min_epochs, avg_interval, epoch_count):  # ✓
        if epoch_count >= min_epochs:
            # Activate early stopping if validation error starts increasing
            continue_train = np.mean(valid_loss_list[-avg_interval:]) < \
                             np.mean(valid_loss_list[-2*avg_interval:-avg_interval])
        else:
            continue_train = True

        return continue_train

    @staticmethod
    def post_train_print(log, epoch_count, final_train_rmse, final_valid_rmse):  # ✓
        # Print training and validation RMSEs to console and write to file
        print("Total number of epochs = ", epoch_count)
        print("Final average training RMSE per data point per bij comp = ",
              final_train_rmse)
        print("Final average validation RMSE per data point per bij comp = ",
              final_valid_rmse)

        with open(log, "a") as write_file:
            print("Total number of epochs = ", epoch_count, file=write_file)
            print("Final average training RMSE per data point per bij comp = ",
                  final_train_rmse, file=write_file)
            print("Final average validation RMSE per data point per bij comp = ",
                  final_valid_rmse, file=write_file)

    @staticmethod
    def test_rmse_ops(y_pred, test_loader, log):  # ✓
        # Calculate testing RMSE, then print to console and write to file
        assert y_pred.shape[0] == len(test_loader.dataset), \
            "Number of rows in y_pred and test loader are different"
        assert y_pred.shape[1] == 9, \
            "Number of columns in y_pred does not equal the number of bij components"

        y = test_loader.dataset.tensors[2].detach().cpu().numpy()
        test_rmse = np.sqrt(np.mean(np.square(y - y_pred)))
        print("Testing RMSE per data point per bij comp = ", test_rmse)

        with open(log, "a") as write_file:
            print("Testing RMSE per data point per bij comp = ", test_rmse,
                  file=write_file)

        return test_rmse
