#!/usr/bin/python

"""Script to write tfrecords for action recognition - Kinetics"""

from random import shuffle
import glob
import cv2
import tensorflow as tf
import numpy as np
import sys
import os
from tqdm import tqdm
import imageio
from video_utils import *

def init(subset):
    shuffle_data = True  # shuffle the addresses before saving
    act_paths = glob.glob('/media/data_cifs/cluster_projects/action_recognition/ActivityNet/Crawler/Kinetics/%s/*'%(subset)) #Load paths to videos organized in folders named by activities
    video_paths = []
    _LABEL_MAP_PATH = 'data/label_map.txt'
    CLASSES_KIN = [x.strip() for x in open(_LABEL_MAP_PATH)]

    for act in act_paths:
        video_paths.extend(glob.glob(os.path.join(act,'*')))
    # read addresses and labels from the 'train' folder
    labels = [CLASSES_KIN.index(video.split('/')[-2]) for video in video_paths] #the second last element of split contains action name

    # to shuffle data
    if shuffle_data:
        c = list(zip(video_paths, labels))
        shuffle(c)
        video_paths, labels = zip(*c)
    return video_paths,labels

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecords(data_path,video_paths,action_labels,n_vids_per_batch,subset):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(data_path)
    for i in tqdm(range(len(video_paths)),desc='Writing tf records..'):
        # print how many images are saved every 1000 images
        if (i!=0 and (not i % n_vids_per_batch)):
            print 'Train data: {}/{}'.format(i, len(video_paths))
            sys.stdout.flush()
        # Load the image
        vid,_ = load_video_with_path_cv2(video_paths[i],n_frames=79)
        label = action_labels[i]
        # Create a feature
        feature = {'%s/label'%(subset): _int64_feature(label)}
        #for i in range(vid.shape[0]):
        feature['%s/video'%(subset)] = _bytes_feature(tf.compat.as_bytes(vid.tostring()))
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()

def main():
    subset='val'
    video_paths,labels = init(subset)
    write_tfrecords('/media/data_cifs/cluster_projects/action_recognition/data/%s.tfrecords'%(subset),video_paths[0:10],labels[0:10],100,subset)

if __name__=="__main__":
    main()
