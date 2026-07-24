import torch
from torch import nn

class LinearRegression(nn.Module):
    def __init__(self, n_input):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(n_input, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred