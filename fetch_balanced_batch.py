#!/usr/bin/python

"""Script to write tfrecords for mixed-mice-data
   Separate script since chunks of video are not
   stored in different files and filenames"""

from random import shuffle, sample
import tensorflow as tf
import numpy as np
import pickle
from config import *
import sys
from random import sample
from tqdm import tqdm
from tf_utils import *
from video_utils import *
import os

choice = np.random.choice

L_POSSIBLE_BEHAVIORS = ["drink",
                        "eat",
                        "groom",
                        "hang",
                        "sniff",
                        "rear",
                        "rest",
                        "walk",
                        "eathand"]

data_root = '/media/data_cifs/mice/mice_data_2018'
video_root = '{}/videos'.format(data_root)
label_root = '{}/labels'.format(data_root)

def sample_behavior_batch(batch_size=16):
    '''Sample batch_size behaviors uniformly
       to prevent any kind of class imbalance
       :param batch_size: Number of chunks per
                          batch for action recognition'''
    behaviors = choice(L_POSSIBLE_BEHAVIORS, batch_size)
    return behaviors

def get_video_chunk_inds(behaviors,
                          behav2video,
                          n_unq_vids_per_batch=1):
    '''Sample a set of video chunks randomly from
       behav2video using the array behaviors
       :param behaviors: batch_size number of
                         randomly sampled behaviors
                         present in a batch
       :param behav2video: Dictionary mapping each
                           behavior to video files
                           which are further mapped
                           to frames where 'behavior'
                           occurs
        :param n_unq_vids_per_batch: Number of unique
                                     videos to sample from
                                     for each batch'''

    batch_video_inds = {}
    for behav in behaviors:
        videos_with_behav = behav2video[behav].keys()
        video_fn = choice(videos_with_behav,
                           n_unq_vids_per_batch)[0]
        #Force multiple chunks in a mini-batch to
        #be chosen from different videos
        while batch_video_inds.has_key(video_fn):
            video_fn = choice(videos_with_behav,
                               n_unq_vids_per_batch)[0]
        curr_ind = choice(behav2video[behav][video_fn],
                           n_unq_vids_per_batch)
        batch_video_inds[video_fn] = curr_ind
    return batch_video_inds

def get_video_chunks(batch_video_inds,
                      behaviors,
                      n_frames=16):
    '''Load video chunks for a mini batch of uniformly
       proportional behavior labels.
       :param batch_video_inds: Dictionary mapping video name
                                to the corresponding frames to
                                be loaded from the video
       :param behaviors: List of behaviors that are labeled
                         for the frames in batch_video_inds
       :param n_frames: Number of frames in a chunk (chunk is a
                                                     single element
                                                     in a minibatch)'''
    video_chunks = []
    for video_fn, frame_ind in batch_video_inds.iteritems():
        video_path = '%s/%s'%(video_root, video_fn)
        #Load n_frames frames before the labeled
        #annotation for prior context
        starting_frame = frame_ind - n_frames
        curr_chunk, _ = get_video_chunk_cv2(video_path,
                                         starting_frame,
                                         n_frames,
                                         normalize=False,
                                         dtype=np.uint8)
        video_chunks.append(curr_chunk)
    video_chunks = np.array(video_chunks)
    video_chunks.astype(np.uint8)
    return video_chunks

def fetch_balanced_batch(behav2video,
                           batch_size=16,
                           n_frames=16):
    '''Wrapper to fetch a mini-batch of videos
       with balanced classes and videos sampled
       randomly per sample
       :param batch_size: Size of mini-batch
                          to load
       :param n_frames: Number of frames present
                        in a single sample'''

    behaviors = sample_behavior_batch(batch_size)
    batch_video_inds = get_video_chunk_inds(behaviors,
                                             behav2video)
    video_chunks = get_video_chunks(batch_video_inds,
                                     behaviors,
                                     n_frames=n_frames)
    return video_chunks

def main():
    behav2video = pickle.load(open('pickles/Behavior2Video.p'))
    video_chunks = fetch_balanced_batch(behav2video)

if __name__=='__main__':
    main()
