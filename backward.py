# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 2019
Revised on Sat Feb 8 2020
@author: Junchao Zhang
Training Code for PFNet
Please Cite the following paper:
Junchao Zhang, Jianbo Shao, Jianlai Chen, Degui Yang, Buge Liang, and
Rongguang Liang. PFNet: An unsupervised deep network for polarization
image fusion. Optics Letters(submitted)
"""
import tensorflow as tf
import model as model
import os
import numpy as np
import h5py
import data_augmentation as DA

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
MAX_EPOCH = 30

MODEL_SAVE_PATH = './model_PFNet/'
MODEL_NAME = 'Fusion'

IMG_SIZE = (40, 40)
IMG_CHANNEL = 2

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    window = fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='SAME')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='SAME') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='SAME') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='SAME') - mu1_mu2

    v1 = 2*mu1_mu2+C1
    v2 = mu1_sq+mu2_sq+C1

    value = (v1*(2.0*sigma12 + C2))/(v2*(sigma1_sq + sigma2_sq + C2))

    # sigma1_sq = sigma1_sq/(mu1_sq+0.00000001)
    v = tf.zeros_like(sigma1_sq) + 0.0001
    sigma1 = tf.where(sigma1_sq<0.0001,v,sigma1_sq)
    return value, sigma1

def loss_func(y_,y):
    img1,img2 = tf.split(y_,2,3)
    img3 = img1*0.5 + img2*0.5
    Win = [11,9,7,5,3]
    loss = 0
    for s in Win:
        loss1, sigma1 = SSIM_LOSS(img1, y, s)
        loss2, sigma2 = SSIM_LOSS(img2, y, s)
        r = sigma1 / (sigma1 + sigma2 + 0.0000001)
        tmp = 1 - tf.reduce_mean(r * loss1) - tf.reduce_mean((1 - r) * loss2)
        loss = loss + tmp
    loss = loss/5.0
    loss = loss + tf.reduce_mean(tf.abs(img3-y))*0.1
    return loss


def backward(train_data, train_num):
    with tf.Graph().as_default() as g:
        with tf.name_scope('input'):
            x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNEL])
            y_ = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNEL])
        # forward
        y = model.forward(x)
        # learning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                                   train_num // BATCH_SIZE,
                                                   LEARNING_RATE_DECAY, staircase=True)
        # loss function
        with tf.name_scope('loss'):
            loss = loss_func(y_,y)
        # Optimizer
        with tf.name_scope('train'):
            # Adam
            optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

        # Save model
        saver = tf.train.Saver(max_to_keep=30)
        epoch = 0

        config = tf.ConfigProto(log_device_placement=True)
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()

            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1].split('-')[-2])

            while epoch < MAX_EPOCH:
                max_step = train_num // BATCH_SIZE
                listtmp = np.random.permutation(train_num)
                j = 0
                for i in range(max_step):
                    file = open("loss.txt", 'a')
                    ind = listtmp[j:j + BATCH_SIZE]
                    j = j + BATCH_SIZE
                    xs = train_data[ind, :, :, :]
                    mode = np.random.permutation(8)
                    xs = DA.data_augmentation(xs,mode[0])


                    _, loss_v, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: xs})
                    file.write("Epoch: %d  Step is: %d After [ %d / %d ] training,  the batch loss is %g.\n" % (
                    epoch + 1, step, i + 1, max_step, loss_v))
                    file.close()
                    # print("Epoch: %d  After [ %d / %d ] training,  the batch loss is %g." % (epoch + 1, i + 1, max_step, loss_v))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME + '_epoch_' + str(epoch + 1)),
                           global_step=global_step)
                epoch += 1


if __name__ == '__main__':
    data = h5py.File('.\TrainingData\imdb_40_128.mat')
    input_data = data["inputs"]
    input_npy = np.transpose(input_data)

    print(input_npy.shape)
    train_num = input_npy.shape[0]
    backward(input_npy,  train_num)