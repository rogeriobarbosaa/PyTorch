import torch
from torch import nn

class LogisticRegression(nn.Module):
    def __init_(self, n_input):
        self.linear = nn.Linear(n_input, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        y_pred = self.act(x)
        return y_pred