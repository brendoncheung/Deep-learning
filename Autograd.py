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

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1) # compress to [64, 764]

# performing backpropagation

from torch import optim

print('Before backpropagation: - ', model[0].weight)

# input to the nn, size adjusted for nn

images = images.view(images.shape[0], -1) # compress to [64, 764]

# configuring optimizer to SGD

optimizer = optim.SGD(model.parameters(), lr=0.1)

# zero out the gradient

optimizer.zero_grad()

# getting the logit score

logit = model(images)

# calculating NLLLoss

criterion = nn.NLLLoss()
lost = criterion(logit, labels)

# compute the gradient

lost.backward()
print('Gradient: - \n', model[0].weight.grad)

# back propagate

optimizer.step()

print('After backpropagation: - ', model[0].weight)


# ================================================== #

epochs = 10


for e in range(epochs):
    running_loss = 0

    for image, label in trainloader:

        image = image.view(image.shape[0], -1)
        optimizer.zero_grad()

        logit = model(image)

        loss = criterion(logit, label)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

import helper

images, labels = next(iter(trainloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
helper.view_classify(img.view(1, 28, 28), ps)






























#
















































#
