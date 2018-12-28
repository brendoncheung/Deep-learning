#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 21:50:06 2018

@author: wingsuncheung
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

image = mpimg.imread('data/curved_lane.jpg')

plt.imshow(image)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')

sobel_y = np.array([[-1, -2, -1],
                   [0, 0, 0,],
                   [1, 2, 1]])


sobel_x = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])


filtered_image = cv2.filter2D(gray, -1, sobel_x)
plt.imshow(filtered_image, cmap='gray') 



