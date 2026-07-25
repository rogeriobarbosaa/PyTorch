import torch
from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self, n_input):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        y_pred = self.act(x)
        return y_pred