import matplotlib.pyplot as plt
import numpy as np
import torch
import helper
from torch import nn

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

# pytorch will keep track of your matrix operation, but you will need to enable this tracking feature

x = torch.zeros(1, requires_grad=True)

# example

x = torch.randn((2, 2), requires_grad=True)
y = x**2

# this grad_fn function tells you the last operation done on this tensor

z = y.mean()

# perform backpropagation

z.backward()

# del z / del x

print(x.grad)
print(x/2)

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logits = model(images)
loss = criterion(logits, labels)

print('Before backward pass: \n', model[0].weight.grad)

loss.backward()

print('After backward pass: \n', model[0].weight.grad)

print(model.parameters)























































#
