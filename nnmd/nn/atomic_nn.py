import torch.nn as nn

class AtomicNN(nn.Module):
    """Model implements a multilayer perceprton
    with sigmoid activation function for one atom
    """
    def __init__(self, hidden_nodes: list | int, input_nodes: int):
        """Create an instance of AtomicNN

        Args:
            hidden_nodes (list | int): configuration of hidden layers nodes
            input_nodes (int): inputs amount of NN
        """
        super().__init__()

        self.model = nn.Sequential()
        if isinstance(hidden_nodes, list):
            self.model.append(nn.Linear(input_nodes, hidden_nodes[0]))
            for prev, curr in zip(range(len(hidden_nodes) - 1), range(1, len(hidden_nodes))):
                self.model.append(nn.Linear(hidden_nodes[prev], hidden_nodes[curr]))
                self.model.append(nn.Sigmoid())
            self.model.append(nn.Linear(hidden_nodes[-1], 1))
        else:
            self.model.append(nn.Linear(5, hidden_nodes))
            self.model.append(nn.Sigmoid())
            self.model.append(nn.Linear(hidden_nodes, 1))

    def forward(self, x):
        logits = self.model(x)
        return logits