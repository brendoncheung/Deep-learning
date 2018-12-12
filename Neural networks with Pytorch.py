import matplotlib.pyplot as plt
import numpy as np
import torch
import helper
from torch import nn

N_HIDDEN_1 = 256
N_OUTPUT = 10

def sigmoid_activation(x):
    return 1 / (1 + torch.exp(-x))

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

data_iter = iter(trainloader)
images, labels = data_iter.next()

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')

inputs = images.view(images.shape[0], -1)
print(inputs.shape)

weight1 = torch.randn((inputs.shape[1]), N_HIDDEN_1)
bias_1 = torch.randn(N_HIDDEN_1)

weight2 = torch.randn(N_HIDDEN_1, N_OUTPUT)
bias_2 = torch.randn(N_OUTPUT)

# first layer

h = sigmoid_activation(torch.mm(inputs, weight1) + bias_1)

out = torch.mm(h, weight2) + bias_2

prediction = softmax(out)

class Network(nn.Module):
    def __init__(self):
        super().__init__()

         # the nn.Linear return the Linear object populated with weights
        self.hidden = nn.Linear(784, 265, bias=True)
        self.output = nn.Linear(265, 10, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.hidden(x)
        x = self.softmax(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

model = Network()
print(model)

import torch.nn.functional as F

class Network2(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc_1 = nn.Linear(784, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 10)

    def forward(self, x):

        x = self.fc_1(x)
        x = F.relu(x) # using ReLU instead of sigmoid
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        x = F.softmax(x, dim=1)

        return x

model2 = Network2()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels)
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model2.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)

print("printing weights")
print(model2.fc_1.weight)

# Using nn.Sequential

input_size = 784
hidden_size = [128, 64] # two layers, first one is 128 units, second is 64
output_size = 10

model3 = nn.Sequential(nn.Linear(input_size, hidden_size[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_size[0], hidden_size[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_size[1], input_size),
                        nn.Softmax(dim=1))

from collections import OrderedDict

model4 = nn.Sequential(OrderedDict([
('first_layer_L_T', nn.Linear(input_size, hidden_size[0])),
('first_layer_activiation', nn.ReLU()),
('second_layer_L_T', nn.Linear(hidden_size[0], hidden_size[1])),
('second_layer_activiation', nn.ReLU()),
('final_layer_L_T', nn.Linear(hidden_size[1], output_size)),
('softmax', nn.Softmax(dim=1))]))

print(model3)
print(model4.first_layer_L_T.weight)

criterion = nn.CrossEntropyLoss()

images = images.view(images.shape[0], -1)

logit = model3(images)

loss = criterion(logit, labels)

print(loss)

print(labels.shape)

model5 = nn.Sequential(nn.Linear(input_size, hidden_size[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_size[0], hidden_size[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_size[1], input_size),
                        nn.LogSoftmax(dim=1))

criterion2 = nn.NLLLoss()

logit = model5(images)
loss = criterion2(logit, labels)

print(loss)
print(torch.exp(logit) * 100)




















#
