# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 2019
Revised on Sat Feb 8 2020
@author: Junchao Zhang
Testing Code for PFNet
Please Cite the following paper:
Junchao Zhang, Jianbo Shao, Jianlai Chen, Degui Yang, Buge Liang, and
Rongguang Liang. PFNet: An unsupervised deep network for polarization
image fusion. Optics Letters(submitted)
"""
import tensorflow as tf
import model as model
import numpy as np
import h5py
import scipy.io

MODEL_SAVE_PATH = './model_PFNet/'
IMG_CHANNEL = 2
IMG_SIZE = (480,640)
BATCH_TEST = 100

def test(test_data):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32,[1,
                                       IMG_SIZE[0],
                                       IMG_SIZE[1],
                                       IMG_CHANNEL])

        y = model.forward(x)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt:
                ckpt.model_checkpoint_path = ckpt.all_model_checkpoint_paths[-1]
                saver.restore(sess,ckpt.model_checkpoint_path)
                Num = test_data.shape[0]
                ImgOut = np.zeros([Num, IMG_SIZE[0], IMG_SIZE[1],1], dtype=np.float32)
                for i in range(Num):
                    print(i)
                    img = sess.run(y, feed_dict={x: test_data[i:i + 1, :, :, :]})
                    ImgOut[i, :, :,:] = np.array(img)
                scipy.io.savemat('results.mat', {'mydata': ImgOut})
            else:
                print("No checkpoint is found.")
                return

if __name__=='__main__':
    data = h5py.File('.\TestingData\imtest.mat')
    input_data = data["inputs"]
    input_npy = np.transpose(input_data)
    print(input_npy.shape)

    test(input_npy)

