import torch
import torch.nn as nn

class AtomicNN(nn.Module):

    def __init__(self, r_cutoff, hidden_nodes, input_nodes = 5, learning_rate = 0.5):
        super().__init__()

        self.r_cutoff = r_cutoff

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

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def forward(self, x):
        logits = self.model(x)
        return logits