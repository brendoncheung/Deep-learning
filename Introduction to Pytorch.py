import matplotlib.pyplot as plt
import numpy as np
import torch

def sigmoid_activation(x):
    return 1 / (1 + torch.exp(-x))

torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 5 random normal variables
features = torch.randn((1, 3))

n_input = features.shape[1]
n_hidden = 2
n_output = 1

weight_1 = torch.randn(n_input, n_hidden)
weight_2 = torch.randn(n_hidden,n_output)

bias_1 = torch.randn((1, n_hidden))
bias_2 = torch.randn((1, n_output))

hidden_layer_1_output = sigmoid_activation(torch.mm(sigmoid_activation(torch.mm(features, weight_1) + bias_1), weight_2) + bias_2)
print(hidden_layer_1_output)

a = np.random.rand(4,3)

print(a)
b = torch.from_numpy(a)
print(b)
b.mul_(4)
print(b)
