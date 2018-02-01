#!/usr/bin/python
"""Evalutation script for testing the trained action recognition models on batches of videos"""

import numpy as np
import tensorflow as tf
from video_utils import *
import i3d
from video_utils import *
from tqdm import tqdm
import os
import sys
from tfrecord_reader import get_video_label_tfrecords

_IMAGE_SIZE = 224
_NUM_CLASSES = 400

_SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
CLASSES_KIN = [x.strip() for x in open(_LABEL_MAP_PATH)]

def get_preds_tensor(input_mode='rgb',n_frames=79, batch_size=10):
    """Function to get the predictions tensor, input placeholder and saver object
        :param input_mode: One of 'rgb','flow','two_stream'"""
    if input_mode == 'rgb':
        rgb_variable_map = {}
        input_fr_rgb = tf.placeholder(tf.float32,shape=[batch_size, n_frames, _IMAGE_SIZE, _IMAGE_SIZE, 3],
                                                name="Input_Video_Placeholder")
        with tf.variable_scope('RGB'):
            #Building I3D for RGB-only input
            rgb_model = i3d.InceptionI3d(_NUM_CLASSES,spatial_squeeze=True,
                                                    final_endpoint='Logits')
            rgb_logits,_ = rgb_model(input_fr_rgb,is_training=False,
                                                dropout_keep_prob=1.0)
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0','')] = variable
        rgb_saver = tf.train.Saver(var_list = rgb_variable_map, reshape=True)
        model_predictions = tf.nn.softmax(rgb_logits)
        top_classes = tf.argmax(model_predictions,axis=1)
        return top_classes,model_predictions, input_fr_rgb, rgb_saver
    else:
        print "#TODO: Implement other input modes"

def evaluate_model(n_val_samples,video2label,input_mode='rgb',n_frames=79,batch_size=10):
    """Function to run evaluation of an action recognition model on val split
        :param input_mode: One of 'rgb','flow','two_stream'
        :param n_frames: Number of frames to represent a video
        :param batch_size: Batch size for validation
        :param n_val_samples: Number of samples for validation"""
        #TODO: Implement precision, recall, conf matrix, F1-score

    n_tp,n_fn,n_fp = 0,0,0

    correct_preds = 0
    top_classes,predictions,input_video_ph,saver = get_preds_tensor(input_mode,n_frames,batch_size)
    with tf.Session().as_default() as sess:
        tfrecords_filename = '../data/val_2.tfrecords'
        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=None)
        videos,labels = get_video_label_tfrecords(filename_queue,10)
        init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        if input_mode=='rgb':
            n_iters = n_val_samples/batch_size
            for i in tqdm(range(0,n_iters),desc='Evaluating Kinetics on val set...'):
                saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
                video_frames_rgb, gt_actions = sess.run([videos,labels])
                top_class_batch = sess.run([top_classes], feed_dict = {input_video_ph: video_frames_rgb})
                print_preds_labels(top_class_batch[0],gt_actions)
                correct_preds += list(top_class_batch[0]==gt_actions).count(True)
                print list(top_class_batch[0]==gt_actions).count(True), "correct predictions"
    classification_accuracy = round(float(correct_preds)*100/n_val_samples,3)
    return classification_accuracy

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    print "Working on GPU %s"%(os.environ["CUDA_VISIBLE_DEVICES"])
    video2label = get_video2label('val')
    n_val_samples = len(video2label)
    print "Evaluating on {} samples..".format(n_val_samples)
    acc = evaluate_model(n_val_samples,video2label)
    print '{}% accuracy on {} samples of val set'.format(acc, n_val_samples)

if __name__=="__main__":
    main()
