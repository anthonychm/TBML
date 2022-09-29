import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from calculator import PopeDataProcessor


class DataLoader:
    def __init__(self, seed, x_train, tb_train, y_train, x_valid, tb_valid, y_valid, x_test, tb_test, y_test,
                 batch_size, device):

        def seed_worker(seed):
            torch_seed = torch.initial_seed()
            np.random.seed(torch_seed + seed)
            random.seed(torch_seed + seed)

        g = torch.Generator(device=device)
        g.manual_seed(seed)

        train_dataset = torchdata.TensorDataset(x_train, tb_train, y_train)
        valid_dataset = torchdata.TensorDataset(x_valid, tb_valid, y_valid)
        test_dataset = torchdata.TensorDataset(x_test, tb_test, y_test)
        self.train_loader = torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                 worker_init_fn=seed_worker(seed), generator=g)
        self.valid_loader = torchdata.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                                 worker_init_fn=seed_worker(seed), generator=g)
        self.test_loader = torchdata.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                worker_init_fn=seed_worker(seed), generator=g)

    @staticmethod
    def check_batch(loader):
        dataiter = iter(loader)
        print(next(dataiter))


class NetworkStructure:
    """
    A class to define the layer structure for the neural network
    """
    def __init__(self, num_hid_layers, num_hid_nodes, af, num_inputs=None, num_tensor_basis=None):
        self.num_hid_layers = num_hid_layers  # Number of hidden layers
        self.num_hid_nodes = num_hid_nodes  # Number of nodes per hidden layer
        self.af = af  # non-linearity string conforming to torch.nn activation functions
        self.af_keywords["alpha"] = "1.0"  # Keyword arguments of chosen activation functions
        self.num_inputs = num_inputs  # Number of scalar invariants
        self.num_tensor_basis = num_tensor_basis  # Number of tensors in the tensor basis

    def set_num_inputs(self, num_inputs):  # Called in _check_structure and set to x.shape[-1]
        self.num_inputs = num_inputs
        return self

    def set_num_tensor_basis(self, num_tensor_basis):  # Called in _check_structure and set to tb.shape[1]
        self.num_tensor_basis = num_tensor_basis
        return self

    def clear_af_keywords(self):
        self.af_keywords = {}

    def set_af_keyword(self, key, value):
        if type(key) is not str:
            raise TypeError("Activation function keyword must be a string")
        # all values are stored as strings for later python eval
        if type(value) is not str:
            value = str(value)
        self.af_keywords[key] = value
        return self

    def check_structure(self, x, tb, y):
        """
        Define number of inputs and tensors in tensor basis and check that they're consistent with
        the specified structure
        :param x: Matrix of input features.  Should be num_points X num_features numpy array
        :param tb: Matrix of tensor basis.  Should be num_points X num_tensor_basis X 9 numpy array
        :param y: Matrix of labels.  Should by num_points X 9 numpy array
        """

        # Check that the inputs, tensor basis array, and outputs all have same number of data points
        assert x.shape[0] == y.shape[0], "Mis-matched shapes between inputs and outputs"
        assert x.shape[0] == tb.shape[0], "Mis-matched shapes between inputs and tensors"

        # Define number of inputs and tensors in tensor basis and check that they're consistent with
        # the specified structure
        if self.num_inputs is None:
            self.set_num_inputs(x.shape[-1])
        else:
            if self.num_inputs != x.shape[-1]:
                print("Mis-matched shapes between specified number of inputs and number of features in input array")
                raise Exception

        if self.num_tensor_basis is None:
            self.set_num_tensor_basis(tb.shape[1])
        else:
            if self.num_tensor_basis != tb.shape[1]:
                print("Mis-matched shapes between specified number of tensors in \
                 tensor basis and number of tensors in tb")
                raise Exception

        # Ensure length of hidden nodes vector is == number of hidden layers
        assert len(self.num_hid_nodes) == self.num_hid_layers


class Tbnn(nn.Module):
    def __init__(self, structure=None, seed=1, weight_init_name="dummy", weight_init_params="dummy"):
        super(Tbnn, self).__init__()
        if structure is None:
            raise TypeError("Network structure not defined")
        self.structure = structure
        self.network = None
        self.seed = seed
        self.weight_init_name = weight_init_name
        weight_init_params.split("=")
        weight_init_params.split(", ")
        self.weight_init_params = weight_init_params

        # Create PyTorch network (try module.append if .add_module does not work)
        self.net = nn.Sequential()
        self.net.add_module("layer1", nn.Linear(self.structure.num_inputs, self.structure.num_hid_nodes[0]))
        self.net.add_module("af1", getattr(nn, self.structure.af[0])(self.structure.af_keywords))
        for layer in range(1, self.structure.num_hid_layers):
            self.net.add_module("layer"+str(layer+1), nn.Linear(self.structure.num_hid_nodes[layer-1],
                                                                self.structure.num_hid_nodes[layer]))
            self.net.add_module("af" + str(layer+1), getattr(nn, self.structure.af[layer])(self.structure.af_keywords))
        self.net.add_module("output_layer", nn.Linear(self.structure.num_hid_nodes[-1], self.structure.num_tensor_basis))
        print("Building of NN complete")

        # Create initialization arguments dictionary and initialize weights and biases
        param_keys, param_val = [], []
        for param in range(0, len(self.weight_init_params)-1):
            param_keys = param_keys.append(weight_init_params[param*2])
            param_val = param_val.append(weight_init_params[(param*2)+1])
        weight_init_dict = dict(zip(param_keys, param_val))

        def init_w_and_b(m):
            torch.manual_seed(seed=seed)
            if isinstance(m, nn.Linear):
                init = getattr(nn.init, self.weight_init_name)
                init(m.weight, **weight_init_dict)
                m.bias.data.fill_(0.01)  # add bias fill as a frontend variable?

        self.net.apply(init_w_and_b)
        print("Weights and biases initialized")

    # Forward propagation
    def forward(self, x, tb):
        coeffs = self.net(x)
        outputs = np.full((len(coeffs), 9), np.nan)
        for bij_comp in range(0, 9):
            outputs[:, bij_comp] = np.multiply(np.tile(coeffs, (10, 1)), tb[:, bij_comp])
        return outputs


class TbnnTVT:
    def __init__(self, loss, optimizer, init_lr, lr_scheduler, lr_scheduler_params, min_epochs, max_epochs, interval, avg_interval, model, print_freq,
                 log):
        self.criterion = getattr(torch.nn, loss)()
        self.optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=init_lr)
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.interval = interval
        self.avg_interval = avg_interval
        self.print_freq = print_freq
        self.log = log

        # Initialise learning rate scheduler
        lr_scheduler_params.split("=")
        lr_scheduler_params.split(", ")
        param_keys, param_val = [], []
        for param in range(0, len(lr_scheduler_params) - 1):
            param_keys = param_keys.append(lr_scheduler_params[param * 2])
            param_val = param_val.append(lr_scheduler_params[(param * 2) + 1])
        lr_param_dict = dict(zip(param_keys, param_val))
        self.lr_scheduler = getattr(torch.optim.lr_scheduler, lr_scheduler)(self.optimizer, **lr_param_dict)

    def fit(self, device, train_loader, valid_loader, model):
        epoch_count = 0
        valid_loss_list = []
        continue_train = True

        # Training loop
        print("Epoch    Time    Average training RMSE per batch    Average validation RMSE per batch")
        while continue_train is True:
            start_time = time.time()
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
            if epoch_count < self.min_epochs:
                continue_train = True

            if epoch_count > self.max_epochs:
                continue_train = False

            # Print average training and validation RMSEs per batch in console
            if epoch_count % self.print_freq == 0:
                print(epoch_count, time.time() - start_time, np.sqrt(avg_train_loss), np.sqrt(avg_valid_loss))

            # Update learning rate
            self.lr_scheduler.step()

        # Output final training and validation RMSEs per batch
        final_train_rmse = np.sqrt(avg_train_loss)
        final_valid_rmse = np.sqrt(avg_valid_loss)
        self.post_train_print(self.log, epoch_count, final_train_rmse, final_valid_rmse)

        return epoch_count, final_train_rmse, final_valid_rmse

    def train_one_epoch(self, train_loader, criterion, optimizer, device, model, epoch_count):
        running_train_loss = 0

        for i, (x, tb, y) in enumerate(train_loader):
            x, tb, y = self.to_device(x, tb, y, device)
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(x, tb)
            loss = criterion(outputs, y)

            # Backward propagation
            loss.backward()
            optimizer.step()

            # Record loss
            running_train_loss += loss.item()
        avg_train_loss = running_train_loss / len(train_loader) # average loss per batch

        return epoch_count, avg_train_loss

    def perform_valid(self, valid_loader, device, model, criterion):
        # Predict on validation data
        running_valid_loss = 0
        with torch.no_grad():
            model.eval()
            for i, (x, tb, y) in enumerate(valid_loader):
                x, tb, y = self.to_device(x, tb, y, device)
                valid_outputs = model(x, tb)
                valid_loss = criterion(valid_outputs, y)
                running_valid_loss += valid_loss

        avg_valid_loss = running_valid_loss / len(valid_loader) # average loss per batch

        return avg_valid_loss

    def perform_test(self, device, enforce_realiz, num_realiz_its, log, test_loader, model):
        # Predict on test data
        with torch.no_grad():
            model.eval()
            for x, tb, y in test_loader:
                x, tb, y = self.to_device(x, tb, y, device)
                y_pred = model(x, tb)

        # Enforce realizability
        if enforce_realiz:
            for i in range(num_realiz_its):
                y_pred = PopeDataProcessor.make_realizable(y_pred)

        # Calculate, print and write testing RMSE
        test_rmse = self.test_rmse_ops(y_pred, y, log)

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
        print("Final average training RMSE per batch = ", final_train_rmse)
        print("Final average validation RMSE per batch = ", final_valid_rmse)

        with open(log, "a") as write_file:
            print("Total number of epochs = ", epoch_count, file=write_file)
            print("Final average training RMSE per batch = ", final_train_rmse, file=write_file)
            print("Final average validation RMSE per batch = ", final_valid_rmse, file=write_file)

    @staticmethod
    def test_rmse_ops(y_pred, y, log):
        # Calculate testing RMSE, then print to console and write to file
        assert y_pred.shape == y.shape, "shape mismatch"
        test_rmse = np.sqrt(np.mean(np.square(y - y_pred)))
        print("Testing RMSE per datapoint = ", test_rmse)

        with open(log, "a") as write_file:
            print("Testing RMSE per datapoint = ", test_rmse, file=write_file)

        return test_rmse
