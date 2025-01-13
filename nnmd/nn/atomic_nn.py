import torch
import torch.nn as nn

class AtomicNN(nn.Module):
    """Model implements a multilayer perceprton
    with sigmoid activation function for one atom
    """
    def __init__(self, hidden_size: list[int] | int, input_size: int):
        """Create an instance of AtomicNN

        Args:
            hidden_size (list[int] | int): configuration of hidden layers nodes
            input_size (int): inputs amount of NN
        """
        super().__init__()

        self.model = nn.Sequential()
        if isinstance(hidden_size, list):
            self.model.append(nn.Linear(input_size, hidden_size[0]))
            for prev, curr in zip(range(len(hidden_size) - 1), range(1, len(hidden_size))):
                self.model.append(nn.Linear(hidden_size[prev], hidden_size[curr]))
            self.model.append(nn.Linear(hidden_size[-1], 1))
        elif isinstance(hidden_size, int):
            self.model.append(nn.Linear(input_size, hidden_size))
            self.model.append(nn.Linear(hidden_size, 1))
        else:
            raise ValueError("hidden_nodes must be a list or an integer")

    def forward(self, x):
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i != len(self.model) - 1:
                x = torch.sigmoid(x)
        return x