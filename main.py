# This is a sample Python script.

# Press Alt+Shift+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

import torch

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


# import scipy

# from scipy.optimize import linear_sum_assignment


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+Shift+B to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    degree = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    threshold = torch.tensor([0.8, 0.8, 0.5])
    print(torch.all(degree > threshold, dim=1))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
