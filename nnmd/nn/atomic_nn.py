import torch
import torch.nn as nn

class AtomicNN(nn.Module):
    """Model implements a multilayer perceprton
    with sigmoid activation function for one atom
    """
    def __init__(self, input_size: int, hidden_sizes: list[int] | int, output_size: int = 1):
        """Create an instance of AtomicNN

        Args:
            hidden_size (list[int] | int): configuration of hidden layers nodes
            input_size (int): inputs amount of NN
        """
        super().__init__()

        self.model = nn.Sequential()
        if isinstance(hidden_sizes, list):
            self.model.append(nn.Linear(input_size, hidden_sizes[0]))
            for prev, curr in zip(range(len(hidden_sizes) - 1), range(1, len(hidden_sizes))):
                self.model.append(nn.Linear(hidden_sizes[prev], hidden_sizes[curr]))
            self.model.append(nn.Linear(hidden_sizes[-1], output_size))
        elif isinstance(hidden_sizes, int):
            self.model.append(nn.Linear(input_size, hidden_sizes))
            self.model.append(nn.Linear(hidden_sizes, output_size))
        else:
            raise ValueError("hidden_sizes must be a list or an integer")

    def forward(self, x):
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i != len(self.model) - 1:
                x = torch.sigmoid(x)
        return x