# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 2019
Revised on Sat Feb 8 2020
@author: Junchao Zhang
Mode Code for PFNet
Please Cite the following paper:
Junchao Zhang, Jianbo Shao, Jianlai Chen, Degui Yang, Buge Liang, and
Rongguang Liang. PFNet: An unsupervised deep network for polarization
image fusion. Optics Letters(submitted)
"""

import tensorflow as tf
import numpy as np

def Dense_block(x):
    for i in range(3):
        shape = x.get_shape().as_list()
        w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9.0 / shape[-1]))
        t = tf.layers.conv2d(x,16,3,(1,1),padding='SAME',kernel_initializer=w_init)
        t = tf.nn.relu(t)
        x = tf.concat([x,t],3)
    return x

def conv_layer(x,filternum,filtersize=3,isactiv=True):
    shape = x.get_shape().as_list()
    w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / filtersize/filtersize/ shape[-1]))
    t = tf.layers.conv2d(x, filternum, filtersize, (1, 1), padding='SAME', kernel_initializer=w_init)
    if isactiv:
        t = tf.nn.relu(t)
    return t

def forward(x):
    S0,DoLP = tf.split(x,2,3)
    w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9.0))
    output = tf.layers.conv2d(S0, 16, 3, (1, 1), padding='SAME', kernel_initializer=w_init)
    output = tf.nn.relu(output)
    Feature_S0 = Dense_block(output)

    w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9.0))
    output = tf.layers.conv2d(DoLP, 16, 3, (1, 1), padding='SAME', kernel_initializer=w_init)
    output = tf.nn.relu(output)
    Feature_DoLP = Dense_block(output)

    Features_fused = tf.concat([Feature_S0,Feature_DoLP],3)

    output = conv_layer(Features_fused, 128, 3)
    output = conv_layer(output,64, 3)
    output = conv_layer(output, 32, 3)
    output = conv_layer(output, 16, 3)
    output = conv_layer(output, 1,3,False)

    return output