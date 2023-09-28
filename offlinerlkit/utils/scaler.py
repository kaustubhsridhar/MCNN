import numpy as np
import os.path as path
import torch


class StandardScaler(object):
    def __init__(self, mu=None, std=None, abs_max=None, use_abs_max=False, device="cpu"):
        self.mu = mu
        self.std = std
        self.abs_max = abs_max
        self.use_abs_max = use_abs_max
        self.device = device

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.abs_max = np.max(np.abs(data), axis=0, keepdims=True)
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

        self.abs_max_tensor = torch.from_numpy(self.abs_max).float().to(self.device)
        self.mu_tensor = torch.from_numpy(self.mu).float().to(self.device)
        self.std_tensor = torch.from_numpy(self.std).float().to(self.device)

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data) / self.abs_max if self.use_abs_max else (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.abs_max * data if self.use_abs_max else self.std * data + self.mu
    
    def save_scaler(self, save_path):
        mu_path = path.join(save_path, "mu.npy")
        std_path = path.join(save_path, "std.npy")
        abs_max_path = path.join(save_path, "abs_max.npy")
        np.save(mu_path, self.mu)
        np.save(std_path, self.std)
        np.save(abs_max_path, self.abs_max)
    
    def load_scaler(self, load_path):
        mu_path = path.join(load_path, "mu.npy")
        std_path = path.join(load_path, "std.npy")
        abs_max_path = path.join(load_path, "abs_max.npy")
        self.mu = np.load(mu_path)
        self.std = np.load(std_path)
        self.abs_max = np.load(abs_max_path)

        self.abs_max_tensor = torch.from_numpy(self.abs_max).float().to(self.device)
        self.mu_tensor = torch.from_numpy(self.mu).float().to(self.device)
        self.std_tensor = torch.from_numpy(self.std).float().to(self.device)

    def transform_tensor(self, data: torch.Tensor):
        device = data.device
        data = self.transform(data.cpu().numpy())
        data = torch.tensor(data, device=device)
        return data
    
    def transform_tensor_with_tensor(self, data: torch.Tensor):
        return (data) / self.abs_max_tensor if self.use_abs_max else (data - self.mu_tensor) / self.std_tensor
    
    def inverse_transform_tensor_with_tensor(self, data: torch.Tensor):
        return self.abs_max_tensor * data if self.use_abs_max else self.std_tensor * data + self.mu_tensor