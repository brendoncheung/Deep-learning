#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:45:23 2018

@author: wingsuncheung
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

img_path = 'data/udacity_sdc.png'

bgr_img = cv2.imread(img_path)
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

gray_img = gray_img.astype("float32")/255

plt.imshow(gray_img, cmap='gray')
plt.show()

filter_vals = np.array([[-1, -1, 1, 1],
                       [-1, -1, 1, 1],
                       [-1, -1, 1, 1],
                       [-1, -1, 1, 1]])

print('Filter shape: ', filter_vals.shape)

filter_1 = filter_vals
filter_2 = -filter_vals
filter_3 = filter_1.T
filter_4 = -filter_3

filters = np.array([filter_1, filter_2, filter_3, filter_4])

class Net(nn.Module):
    
    def __init__(self, weight):
        super(Net, self).__init__()
        
        k_height, k_width = weight.shape[2:]
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)

        pooled_x = self.pool(activated_x)
        
        return conv_x, activated_x, pooled_x
    
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)

print(model)

def viz_layer(layer, n_filter=4):
    fig = plt.figure(figsize=(20, 20))
    
    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filter, i+1)
        ax = imshow(np.squeeze(layer[0, i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i+1))
































