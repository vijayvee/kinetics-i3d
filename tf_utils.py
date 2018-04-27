#!/usr/bin/python
"""Bundling all tensorflow utilities in tf_utils.py"""
import numpy as np
import tensorflow as tf
from video_utils import *
import i3d
import pickle
from tqdm import tqdm
import os
import sys
import random
from tfrecord_reader import get_video_label_tfrecords
from test_batch_videos import evaluate_model
from time import gmtime, strftime

_IMAGE_SIZE = 224
_NUM_CLASSES = 9

_SAMPLE_VIDEO_FRAMES = 16
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'mice': 'ckpt_dir/Mice_ACBM_I3D_0.0001_adam_10_12000_2018_03_09_19_53_57.ckpt.meta',
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
CLASSES_KIN = [x.strip() for x in open(_LABEL_MAP_PATH)]
CLASSES_MICE = ["drink", "eat", "groom", "hang", "sniff", "rear", "rest", "walk", "eathand"]

def get_filter_conv3d(name, shape, trainable):
    """Function to create a 3d conv kernel"""
    conv_w = tf.get_variable(name,
                        shape=shape,
                        dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        trainable=trainable)
    return conv_w

def conv3d(name, input, shape):
    with tf.variable_scope(name):
        conv_w = get_filter_conv3d('conv_1x1', shape, trainable=True)
        conv = tf.nn.conv3d(input=input, filter=conv_w,
                            strides=[1,1,1,1,1],
                            padding='SAME')
        return conv

def get_loss(predictions, ground_truth):
    """Function to get the loss tensor for I3d
        :param predictions: Tensor with a batch of I3D action predictions
        :param ground_truth: Tensor with the ground truth for predictions"""
    class_weights = pickle.load(open('class_weights_all_mice_2018.p'))
    class_weights = tf.constant([float(class_weights[cls]) for cls in CLASSES_MICE])
    weights = tf.reduce_sum(class_weights*ground_truth,axis=1)
    unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=ground_truth)
    weighted_loss = unweighted_loss*weights
    print "Unweighted loss shape: {} Weighted loss shape: {}".format(unweighted_loss.shape, weighted_loss.shape)
    return tf.reduce_mean(weighted_loss)


def get_preds_loss(ground_truth, input_mode='rgb',
                    n_frames=16, num_classes=_NUM_CLASSES,
                    batch_size=10, dropout_keep_prob=0.6):
    """Function to get the predictions tensor,
        loss, input placeholder and saver object
        :param ground_truth: Tensor to hold ground truth
        :param input_mode: One of 'rgb','flow','two_stream'"""
    rgb_variable_map = {}
    input_fr_rgb = tf.placeholder(tf.float32,
                                    shape=[batch_size,
                                           n_frames,
                                           _IMAGE_SIZE, _IMAGE_SIZE,
                                           3],
                                    name='Input_Video_Placeholder')
    with tf.variable_scope('RGB'):
        #Building I3D for RGB-only input
        rgb_model = i3d.InceptionI3d(spatial_squeeze=True,
                                       final_endpoint='Mixed_5c')
        rgb_mixed_5c,_ = rgb_model(input_fr_rgb,
                                     is_training=False,
                                     dropout_keep_prob=1.0)

        with tf.variable_scope('Logits_Mice'):
            net = tf.nn.avg_pool3d(rgb_mixed_5c,
                                    ksize=[1, 2, 7, 7, 1],
                                    strides=[1, 1, 1, 1, 1],
                                    padding='VALID')

            net = tf.nn.dropout(net, dropout_keep_prob)
            logits = conv3d(name='Logits',input=net,
                              shape=[1,1,1,1024,
                              num_classes])
            logits = tf.squeeze(logits,
                                  [2, 3],
                                  name='SpatialSqueeze')
            averaged_logits = tf.reduce_mean(logits, axis=1)

    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB' and 'Logits' not in variable.name:
            rgb_variable_map[variable.name.replace(':0','')] = variable
    rgb_saver = tf.train.Saver(var_list = rgb_variable_map,
                                reshape=True)
    model_predictions = tf.nn.softmax(averaged_logits)
    top_classes = tf.argmax(model_predictions,axis=1)
    loss = get_loss(model_predictions, ground_truth)
    return model_predictions, loss, top_classes, input_fr_rgb, rgb_saver

def get_optimizer(loss, optim_key='adam', learning_rate=1e-4, momentum=0.9):
    """Function to return an optimizer
        :param loss: Tensor with the loss for action recognition
        :param optim_key: The type of optimizer to be used
        :param learning_rate: Learning rate to use for optimizing"""
    if optim_key=='adam':
        optim = tf.train.AdamOptimizer(
                            learning_rate=learning_rate
                            )
    elif optim_key=='momentum':
        optim = tf.train.MomentumOptimizer(
                            learning_rate=learning_rate,
                            momentum=momentum
                            )
    elif optim_key=='sgd':
        optim = tf.train.GradientDescentOptimizer(
                            learning_rate=learning_rate
                            )
    step = optim.minimize(loss)
    return step
