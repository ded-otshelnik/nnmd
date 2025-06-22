import torch
import torch.nn as nn


class AtomicNN(nn.Module):
    """Model implements a multilayer perceptron
    for one species with a single output for one atom.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | int,
        output_size: int = 1,
        activation: str = torch.sigmoid,
    ):
        """Create an instance of AtomicNN

        Args:
            input_size (int): inputs amount of NN
            hidden_sizes (list[int] | int): configuration of hidden layers nodes
            output_size (int): outputs amount of NN
            activation (str): activation function to use in hidden layers
        """
        super().__init__()

        self.model = nn.Sequential()
        if isinstance(hidden_sizes, list):
            self.model.append(nn.Linear(input_size, hidden_sizes[0]))
            for prev, curr in zip(
                range(len(hidden_sizes) - 1), range(1, len(hidden_sizes))
            ):
                self.model.append(nn.Linear(hidden_sizes[prev], hidden_sizes[curr]))
            self.model.append(nn.Linear(hidden_sizes[-1], output_size))
        elif isinstance(hidden_sizes, int):
            self.model.append(nn.Linear(input_size, hidden_sizes))
            self.model.append(nn.Linear(hidden_sizes, output_size))
        else:
            raise ValueError("hidden_sizes must be a list or an integer")

        self.activation = activation

    def forward(self, x):
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i != len(self.model) - 1:
                x = self.activation(x)
        return x
