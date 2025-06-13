import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

        self.dtype = torch.float32
        self.device = torch.device("cpu")

        # utils
        self.l1_loss = F.l1_loss
        self.mse_loss = F.mse_loss
        self.huber_loss = F.smooth_l1_loss

    def print_parameter_devices(self, model):
        for name, param in model.named_parameters():
            print(f"{name}: {param.device}")

    def to_device(self, device):
        self.device = device
        self.to(device)

    def preprocess_state(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Preprocess the state to the required format.
        """
        if isinstance(state, torch.Tensor):
            state = state.to(self.device)
        elif isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device).to(self.dtype)
        else:
            raise ValueError("Unsupported state type. Must be a tensor or numpy array.")

        if len(state.shape) == 1 or len(state.shape) == 3:
            state = state.unsqueeze(0)
        if len(state.shape) > 3:
            state = state.view(state.size(0), -1)

        return state

    def compute_gradient_norm(self, models, names, device, dir="None", norm_type=2):
        grad_dict = {}
        for i, model in enumerate(models):
            if model is not None:
                total_norm = torch.tensor(0.0, device=device)
                try:
                    for param in model.parameters():
                        if (
                            param.grad is not None
                        ):  # Only consider parameters that have gradients
                            param_grad_norm = torch.norm(param.grad, p=norm_type)
                            total_norm += param_grad_norm**norm_type
                except:
                    try:
                        param_grad_norm = torch.norm(model.grad, p=norm_type)
                    except:
                        param_grad_norm = torch.tensor(0.0)
                    total_norm += param_grad_norm**norm_type

                total_norm = total_norm ** (1.0 / norm_type)
                grad_dict[dir + "/grad/" + names[i]] = total_norm.item()

        return grad_dict

    def compute_weight_norm(self, models, names, device, dir="None", norm_type=2):
        norm_dict = {}
        for i, model in enumerate(models):
            if model is not None:
                total_norm = torch.tensor(0.0, device=device)
                try:
                    for param in model.parameters():
                        param_norm = torch.norm(param, p=norm_type)
                        total_norm += param_norm**norm_type
                except:
                    param_norm = torch.norm(model, p=norm_type)
                    total_norm += param_norm**norm_type
                total_norm = total_norm ** (1.0 / norm_type)
                norm_dict[dir + "/weight/" + names[i]] = total_norm.item()

        return norm_dict

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values and counts for each key
        sum_dict = {}
        count_dict = {}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                if key not in sum_dict:
                    sum_dict[key] = 0
                    count_dict[key] = 0
                sum_dict[key] += value
                count_dict[key] += 1

        # Calculate the average for each key
        avg_dict = {key: sum_val / count_dict[key] for key, sum_val in sum_dict.items()}

        return avg_dict

    def flat_grads(self, grads: tuple):
        """
        Flatten the gradients into a single tensor.
        """
        flat_grad = torch.cat([g.view(-1) for g in grads])
        return flat_grad
