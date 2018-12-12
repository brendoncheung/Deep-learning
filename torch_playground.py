import matplotlib.pyplot as plt
import numpy
import torch
from torch import nn

class Car:

    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0

    def get_descriptive_name(self):
        long_name = str(self.year) + ' ' + self.make + ' ' + self.model
        return long_name.title()

class ElectricCar(Car):
    def __init__(self, make, model, year):
        super().__init__(make, model, year)

m = nn.Linear(1, 1)
# input = torch.randn(128, 20)
# output = m(input)

x = torch.randn((2,2), requires_grad=True)

y = x ** 2

z = y**3

z.backward(torch.ones((2,2)))

print(x.grad)
print(6 * x * y**2)







































#
