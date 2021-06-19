import torch
from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, class_num):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(input_dim, class_num)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        x = self.lr(x)
        x = self.sm(x)
        return x
