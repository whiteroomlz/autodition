import torch

ACTIVATIONS_MAPPING = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
    "none": torch.nn.Identity,
}
