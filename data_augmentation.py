# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:09:45 2018

@author: Junchao Zhang
"""
import numpy as np

def data_augmentation(imagein, mode):
    image = np.transpose(imagein,(1,2,3,0))
    if mode == 0:
        # original
        return imagein
    elif mode == 1:
        # flip up and down
        image = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        image = np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        image = np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        image = np.flipud(image)
    imageout = np.transpose(image,(3,0,1,2))
    return imageout