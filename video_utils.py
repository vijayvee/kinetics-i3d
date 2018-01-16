#!/usr/bin/python
import imageio
import numpy as np
import cv2
from random import sample
import pandas as pd
import glob

"""Script for video related utilities"""

BATCH_SIZE = 10
N_FRAMES = 79
IMAGE_SIZE = 224
NUM_CLASSES=400
KINETICS_ROOT = '/media/data_cifs/cluster_projects/action_recognition/ActivityNet/Crawler/Kinetics'
subset = 'train'
VIDEOS_ROOT = KINETICS_ROOT + '/' + subset
_LABEL_MAP_PATH = 'data/label_map.txt'
CLASSES_KIN = [x.strip() for x in open(_LABEL_MAP_PATH)]
video2label = {}

def read_video(video_id,label,n_frames,subset):
    """Function to read a single video given by video_fn and return n_frames equally spaced frames from the video
        video_fn: Filename of the video to read
        n_frames: Number of frames to read in the video"""
    VIDEOS_ROOT = KINETICS_ROOT + '/' + subset
    video_fn = glob.glob("%s/%s/%s*mp4"%(VIDEOS_ROOT,label,video_id))
    if video_fn == []:
        return np.zeros((224,224,3)),(224,224,3)
    vid = imageio.get_reader(video_fn[0],'ffmpeg')
    ind_frames = map(int,np.linspace(0,vid.get_length()-1,n_frames))
    curr_frames = [vid.get_data(i) for i in ind_frames]
    resized_frames = [cv2.resize(i,(224,224)) for i in curr_frames]
    curr_frames = np.array(resized_frames)
    norm_frames = (curr_frames/127.5) - 1.
    return norm_frames,norm_frames.shape

def get_video2label(subset):
    """Function to load a mapping from video id to label"""
    subset_csv = KINETICS_ROOT + '/kinetics_{}.csv'.format(subset)
    df = pd.read_csv(subset_csv)
    video_ids, labels = df.youtube_id, df.label
    video2label = {v:l for v,l in zip(video_ids,labels)}
    return video2label

def get_video_batch(video2label,batch_size=BATCH_SIZE,validation=False,val_ind=0,n_frames=N_FRAMES,class_index=True):
    """Function to return a random batch of videos for train mode and a specific set of videos for val mode.
        :param batch_size: Specifies the size of the batch of videos to be returned
        :param validation: Flag to specify training mode (True for val phase)
        :param val_ind: Index of the batch of val videos to retrieve"""
    if validation:
        curr_videos = video2label.keys()[val_ind:val_ind+batch_size]
        curr_labels = [video2label[v] for v in curr_videos]
        video_rgb_frames = [read_video(curr_vid,curr_label,n_frames,'val')[0] for curr_vid,curr_label in zip(curr_videos,curr_labels)]
        import ipdb; ipdb.set_trace()
        if class_index:
            curr_labels = [CLASSES_KIN.index(action) for action in curr_labels]
        return np.array(video_rgb_frames),np.array(curr_labels)
    else:
        curr_inds = sample(0,len(video2label)-1,batch_size)
        curr_videos = video2label.keys()[curr_inds]
        curr_labels = [video2label[v] for v in curr_videos]
        video_rgb_frames = [read_video(curr_vid,curr_label,n_frames,'train')[0] for curr_vid,curr_label in zip(curr_videos,curr_labels)]
        if class_index:
            curr_labels = [CLASSES_KIN.index(action) for action in curr_labels]
        return np.array(video_rgb_frames),np.array(curr_labels)
