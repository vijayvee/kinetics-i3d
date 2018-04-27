#!/usr/bin/python
"""Script to train videos in batches"""

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
from tf_utils import *

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

def validation_accuracy(n_val_samples,video2label,
                          sess,input_video_ph,batch_size,
                          top_classes,val_tfrecords):
    """Function to compute accuracy on a validation set
        :param n_val_samples: Number of samples to validate on
        :param video2label: Dictionary mapping video ids to labels
        :param sess: Session reference for the current set of trained weights
        :param input_video_ph: Placeholder for input batch of videos
        :param batch_size: Batch size used during training
        :param ground_truth_ph: Ground truth placeholder for action recognition
        :param top_classes: Tensor holding the batch top class predictions
        :param val_tfrecords: Tfrecords filename for validation set"""

    correct_preds = 0
    tfrecords_filename = val_tfrecords
    filename_queue = tf.train.string_input_producer([tfrecords_filename],
                                                      num_epochs=None)
    videos,labels = get_video_label_tfrecords(filename_queue,batch_size)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    n_iters = n_val_samples/batch_size
    for i in tqdm(range(0,n_iters),desc='Computing validation set accuracy...'):
        video_frames_rgb, gt_actions = sess.run([videos,labels])
        top_class_batch = sess.run([top_classes],
                                    feed_dict = {input_video_ph:
                                                video_frames_rgb})
        correct_preds += list(top_class_batch[0]==gt_actions).count(True)
    classification_accuracy = round(float(correct_preds)*100/n_val_samples,3)
    print "Validation complete! Accuracy: {}".format(classification_accuracy)
    return classification_accuracy


def train_batch_videos(n_train_samples, n_epochs, video2label,
                        input_mode='rgb', save_every=1000,
                        print_every=10, action_every=50,
                        num_classes=9, n_frames=16,
                        batch_size=10, n_val_samples=10000,
                        early_stopping=5, optim_key='adam',
                        tfrecords_filename='../data/train.tfrecords',
                        val_tfrecords='../data/val_2.tfrecords',
                        learning_rate=1e-4):
    """Function to train videos in batches.
        :param n_train_samples: Number of training videos
        :param n_epochs: Number of epochs to train the model
        :param val_accuracy_iter: Interval for checking validation accuracy
        :param video2label: Dictionary mapping video ids to action labels
        :param input_mode: One of 'rgb','flow','two_stream'
        :param save_every: Save checkpoint of weights every save_every iterations
        :param print_every: Print loss log every print_every iterations
        :param action_every: Print action predictions with their respective ground truth every action_every iterations
        :param num_classes: Number of action classes
        :param n_frames: Number of frames to represent a video
        :param batch_size: Batch size for training"""
    correct_preds = 0.
    ground_truth = tf.placeholder(tf.float32,shape=[batch_size,num_classes])
    predictions,loss,top_classes,input_video_ph,saver = get_preds_loss(ground_truth=ground_truth,
                                                                        input_mode=input_mode,
                                                                        n_frames=n_frames,
                                                                        batch_size=batch_size)
    saver_mice = tf.train.Saver()
    step = get_optimizer(loss,optim_key='adam',learning_rate=learning_rate)
    best_val_accuracy = -1.
    val_accuracy_iter = n_train_samples/batch_size
    with tf.Session().as_default() as sess:
        #tfrecords_filename = './data/val_2.tfrecords'
        filename_queue = tf.train.string_input_producer([tfrecords_filename],
                                                          num_epochs=None)
        videos,labels = get_video_label_tfrecords(filename_queue,batch_size,
                                                    subset='train',shuffle=True)
        init_op = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        if input_mode=='rgb':
            n_iters = (n_epochs*n_train_samples)/batch_size
            saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
            for i in tqdm(range(0,n_iters),desc='Training I3D on Kinetics train set...'):
                video_frames_rgb, gt_actions = sess.run([videos,labels])
                if i==0:
                    print "Obtained frames and actions", \
                            video_frames_rgb.shape, gt_actions.shape
                gt_actions_oh = np.eye(num_classes)[gt_actions]
                curr_loss,top_class_batch,_ = sess.run([loss,
                                                        top_classes,
                                                        step],
                                                        feed_dict = {input_video_ph:
                                                                      video_frames_rgb,
                                                                     ground_truth:
                                                                      gt_actions_oh})

                correct_preds += list(top_class_batch==gt_actions).count(True)
                if i%print_every==0:
                    print 'Iteration-{} Current training loss: {} Current training accuracy: {}'.format((i+1),
                                                                                                curr_loss,
                                                                                                round(correct_preds/float((i+1)*batch_size),3))
                if i%action_every==0:
                    print_preds_labels(top_class_batch, gt_actions)
                if i%save_every==0:
		    curr_time = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
                    saver_mice.save(sess,'./ckpt_dir/Mice_ACBM_FineTune_I3D_%s_%s_%s_%s_%s.ckpt'%(learning_rate,optim_key,n_epochs,str(i),curr_time))
    #TODO: Try with mice val set
    print "Training completed with best accuracy: {}".format(best_val_accuracy)
    return best_val_accuracy

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    print "Working on GPU %s"%(os.environ["CUDA_VISIBLE_DEVICES"])
    video2label = get_video2label('val')
    n_train_samples = len(video2label)
    best_val_accuracy = train_batch_videos(n_train_samples=n_train_samples,
                                            n_epochs=10, video2label=video2label,
                                            tfrecords_filename=sys.argv[2],batch_size=16,
                                            val_tfrecords=None,
                                            learning_rate=1e-4)
