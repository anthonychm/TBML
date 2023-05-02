import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as torchdata


def tkenn_ops(x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size,
              num_hid_layers, num_hid_nodes, af, af_params, seed, weight_init,
              weight_init_params, loss, optimizer, init_lr, lr_scheduler,
              lr_scheduler_params, min_epochs, max_epochs, interval, avg_interval,
              print_freq, log, num_inputs):  # ✓

    # Use GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare batch data in dataloaders and check dataloaders ✓
    dataloader = DataLoader(x_train, y_train, x_valid, y_valid, x_test, y_test,
                            batch_size)  # ✓
    DataLoader.check_data_loaders(dataloader.train_loader, x_train, y_train,
                                  num_inputs)  # ✓
    DataLoader.check_data_loaders(dataloader.valid_loader, x_valid, y_valid,
                                  num_inputs)  # ✓
    DataLoader.check_data_loaders(dataloader.test_loader, x_test, y_test,
                                  num_inputs)  # ✓

    # Prepare TKENN architecture structure and check structure ✓
    structure = NetworkStructure(num_hid_layers, num_hid_nodes, af, af_params,
                                 num_inputs)  # ✓
    structure.check_structure()
    print("TKENN architecture structure and data loaders checked")

    # Construct TKENN and perform training, validation and testing ✓
    tkenn = Tkenn(device, seed, structure, weight_init=weight_init,
                  weight_init_params=weight_init_params).double()  # ✓
    tkenn_tvt = TkennTVT(loss, optimizer, init_lr, lr_scheduler, lr_scheduler_params,
                         min_epochs, max_epochs, interval, avg_interval, print_freq,
                         log, tkenn)  # ✓
    epoch_count, final_train_rmse, final_valid_rmse = \
        tkenn_tvt.fit(device, dataloader.train_loader, dataloader.valid_loader, tkenn)  # ✓
    y_pred, test_rmse = tkenn_tvt.perform_test(device, log, dataloader.test_loader,
                                               tkenn)  # ✓

    return epoch_count, final_train_rmse, final_valid_rmse, y_pred, test_rmse


class DataLoader:
    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size):
        # ✓

        # Convert data from np.array into torch.tensor ✓
        self.x_train = torch.from_numpy(x_train)
        self.y_train = torch.from_numpy(y_train)
        self.x_valid = torch.from_numpy(x_valid)
        self.y_valid = torch.from_numpy(y_valid)
        self.x_test = torch.from_numpy(x_test)
        self.y_test = torch.from_numpy(y_test)

        # Run TensorDataset and DataLoader for batch allocation ✓
        train_dataset = torchdata.TensorDataset(self.x_train, self.y_train)
        valid_dataset = torchdata.TensorDataset(self.x_valid, self.y_valid)
        test_dataset = torchdata.TensorDataset(self.x_test, self.y_test)
        # Order of data points already shuffled during train-validation-test splits, so
        # we can set shuffle=False here ✓
        self.train_loader = torchdata.DataLoader(train_dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        self.valid_loader = torchdata.DataLoader(valid_dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        self.test_loader = torchdata.DataLoader(test_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=0)

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
    def check_data_loaders(data_loader, x, y, num_inputs):  # ✓
        # Check that the inputs and bij outputs all have the same number of data points
        assert x.shape[0] == y.shape[0], "Mismatched shapes between inputs and outputs"

        # Check data loader dataset shapes
        shape_error = "Error in data loader dataset shape"
        assert data_loader.dataset.tensors[0].size(dim=0) == x.shape[0], shape_error
        assert data_loader.dataset.tensors[0].size(dim=1) == num_inputs, shape_error
        assert data_loader.dataset.tensors[1].size(dim=0) == y.shape[0], shape_error
        # assert data_loader.dataset.tensors[1].size(dim=1) == 1, shape_error


class NetworkStructure:

    def __init__(self, num_hid_layers, num_hid_nodes, af, af_params, num_inputs):  # ✓
        self.num_hid_layers = num_hid_layers  # Number of hidden layers
        self.num_hid_nodes = num_hid_nodes  # Number of nodes per hidden layer
        self.af = af  # Activation functions
        self.num_inputs = num_inputs  # Number of inputs

        # Create activation function arguments pandas dataframe
        af_df = pd.Series(af_params)
        self.af_df = af_df.str.split(pat=", ")

    def check_structure(self):  # ✓
        # Check that the number of hidden nodes, activation functions and their
        # parameters are consistent with the number of hidden layers
        assert len(self.num_hid_nodes) == self.num_hid_layers, \
            "Mismatch between the length of num_hid_nodes and value of num_hid_layers"
        assert len(self.af) == self.num_hid_layers, \
            "Mismatch between the length of af and value of num_hid_layers"
        assert len(self.af_df) == self.num_hid_layers, \
            "Mismatch between the length of af_params and value of num_hid_layers"


class Tkenn(nn.Module):
    def __init__(self, device, seed, structure=None, weight_init="dummy",
                 weight_init_params="dummy"):  # ✓
        super(Tkenn, self).__init__()
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

        # Create PyTorch neural network ✓ [CHECK IN CODERUN]
        self.net = nn.Sequential()
        layer = 0
        self.net.add_module("layer1", nn.Linear(self.structure.num_inputs,
                                                self.structure.num_hid_nodes[layer]))
        af_param_dict = retrieve_af_params(self.structure.af_df, layer)  # ✓
        self.net.add_module("af1", getattr(nn, self.structure.af[layer])(**af_param_dict))
        for layer in range(1, self.structure.num_hid_layers):
            self.net.add_module("layer"+str(layer+1),
                                nn.Linear(self.structure.num_hid_nodes[layer-1],
                                          self.structure.num_hid_nodes[layer]))
            af_param_dict = retrieve_af_params(self.structure.af_df, layer)  # ✓
            self.net.add_module("af" + str(layer+1),
                                getattr(nn, self.structure.af[layer])(**af_param_dict))
        self.net.add_module("output_layer",
                            nn.Linear(self.structure.num_hid_nodes[-1], 1))
        print("Building of NN complete")

        # Create initialization arguments dictionary and initialize weights and biases ✓
        param_keys, param_val = [], []
        for i, string in enumerate(self.weight_init_params):
            if (i + 2) % 2 == 0:
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
    def forward(self, x):
        y_pred = self.net(x)

        return y_pred


class TkennTVT:
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
            if (i + 2) % 2 == 0:
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
        print("Epoch    Average training RMSE per data point   "
              "Average validation RMSE per data point")
        while continue_train:
            epoch_count += 1
            model.train()
            epoch_count, avg_train_loss \
                = self.train_one_epoch(train_loader, self.criterion, self.optimizer,
                                       device, model, epoch_count)  # ✓

            # Predict on validation data for early stopping ✓
            if epoch_count % self.interval == 0:
                avg_valid_loss = self.perform_valid(valid_loader, device, model,
                                                    self.criterion)  # ✓
                valid_loss_list.append(avg_valid_loss)
                continue_train = self.check_conv(valid_loss_list, self.min_epochs,
                                                 self.avg_interval, epoch_count)  # ✓
                if continue_train is False:
                    break

            # The check_conv method ensures number of epochs is > min epochs
            # The if statement below ensures number of epochs is < max epochs ✓
            if epoch_count > self.max_epochs:
                break

            # Print average training and validation RMSEs per data point in console  # ✓
            if epoch_count % self.print_freq == 0:
                print(epoch_count, np.sqrt(avg_train_loss), np.sqrt(avg_valid_loss))

            # Update learning rate ✓
            self.lr_scheduler.step()

        # Output final training and validation RMSEs per data point ✓
        final_train_rmse = np.sqrt(avg_train_loss)
        final_valid_rmse = np.sqrt(avg_valid_loss)
        self.post_train_print(self.log, epoch_count, final_train_rmse,
                              final_valid_rmse)  # ✓

        return epoch_count, final_train_rmse, final_valid_rmse

    def train_one_epoch(self, train_loader, criterion, optimizer, device, model,
                        epoch_count):  # ✓
        running_train_loss = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = self.to_device(x, y, device)  # ✓

            # Forward propagation ✓
            logk_batch_pred = model(x)  # ✓
            # Calculate MSE per datapoint for entire batch
            loss = criterion(logk_batch_pred, y)

            # Backward propagation ✓
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record loss ✓
            running_train_loss += loss.item()  # sum of MSE for all batches

        # Calculate average loss per data point ✓
        avg_train_loss = running_train_loss / len(train_loader)

        return epoch_count, avg_train_loss

    def perform_valid(self, valid_loader, device, model, criterion):  # ✓
        # Predict on validation data ✓
        running_valid_loss = 0
        with torch.no_grad():
            model.eval()
            for i, (x, y) in enumerate(valid_loader):
                x, y = self.to_device(x, y, device)  # ✓
                valid_outputs = model(x)
                valid_loss = criterion(valid_outputs, y)
                running_valid_loss += valid_loss.item()

        # Calculate average loss per data point ✓
        avg_valid_loss = running_valid_loss / len(valid_loader)

        return avg_valid_loss

    def perform_test(self, device, log, test_loader, model):  # ✓
        # Predict on test data ✓
        with torch.no_grad():
            model.eval()
            for x, y in test_loader:
                x, y = self.to_device(x, y, device)  # ✓
                y_pred_tmp = model(x)
                if "y_pred" in locals():
                    y_pred = torch.vstack((y_pred, y_pred_tmp))
                else:
                    y_pred = y_pred_tmp

        # Convert y_pred to np array ✓
        y_pred = y_pred.detach().cpu().numpy()

        # Calculate, print and write testing RMSE ✓
        test_rmse = self.test_rmse_ops(y_pred, test_loader, log)  # ✓

        return y_pred, test_rmse

    @staticmethod
    def to_device(x, y, device):  # ✓
        x = x.to(device)
        y = y.to(device)
        return x, y

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
        print("Final average training RMSE per data point = ", final_train_rmse)
        print("Final average validation RMSE per data point = ", final_valid_rmse)

        with open(log, "a") as write_file:
            print("Total number of epochs = ", epoch_count, file=write_file)
            print("Final average training RMSE per data point = ", final_train_rmse,
                  file=write_file)
            print("Final average validation RMSE per data point = ", final_valid_rmse,
                  file=write_file)

    @staticmethod
    def test_rmse_ops(y_pred, test_loader, log):  # ✓
        # Calculate testing RMSE, then print to console and write to file
        assert y_pred.shape[0] == len(test_loader.dataset), \
            "Number of rows in y_pred and test loader are different"
        assert y_pred.shape[1] == 1, \
            "Number of columns in y_pred does not equal number of required outputs"

        y = test_loader.dataset.tensors[1].detach().cpu().numpy()
        test_rmse = np.sqrt(np.mean(np.square(y-y_pred)))
        print("Testing RMSE per data point = ", test_rmse)

        with open(log, "a") as write_file:
            print("Testing RMSE per data point = ", test_rmse, file=write_file)

        return test_rmse

