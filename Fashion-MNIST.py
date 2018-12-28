import matplotlib.pyplot as plt
import numpy as np
import torch
import helper
from torch import nn

from torchvision import datasets, transforms

# define the transformer that Normalize

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# download the data Set

trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

images, label = next(iter(trainloader))

# reduce the dimensions

images = images.view(images.shape[0], -1)

print(images.shape)
print(label.shape)

# defining the model

input_features =        images.shape[1]
hidden_layer_units_1 =  128
hidden_layer_units_2 =  64
output_units =          10

epochs = 10

model = nn.Sequential(
nn.Linear(input_features, hidden_layer_units_1),
nn.ReLU(),
nn.Linear(hidden_layer_units_1, hidden_layer_units_2),
nn.ReLU(),
nn.Linear(hidden_layer_units_2, output_units),
nn.LogSoftmax()
)

criterion = nn.NLLLoss()

from torch import optim

optimizer = optim.SGD(model.parameters(), lr=0.01)

for e in range(epochs):

    running_loss = 0

    for images, labels in trainloader:

        optimizer.zero_grad()

        image = images.view(images.shape[0], -1)

        logit = model(image)

        loss = criterion(logit, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    else:

        print(f"Training loss: {running_loss/len(trainloader)}")

import helper

# Test out your network!



def run_classifier():
    
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    img = images[0]
    # Convert 2D image to 1D vector
    img = img.resize_(1, 784)
    
    # TODO: Calculate the class probabilities (softmax) for img
    ps = torch.exp(model(img))
    
    # Plot the image and probabilities
    helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')








































#
