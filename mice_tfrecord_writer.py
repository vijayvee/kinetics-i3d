#!/usr/bin/python

"""Script to write tfrecords for mixed-mice-data
Separate script since chunks of video are not stored in different files and filenames"""

from random import shuffle
import pickle
import h5py
import glob
import cv2
import tensorflow as tf
import numpy as np
import sys
import os
from tqdm import tqdm
import imageio
from video_utils import *
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

def get_lists(subset,ratio):
    labels = '{}/{}_labels_norest.pkl'.format(data_root,subset)
    videos = '{}/{}_videos_norest.pkl'.format(data_root,subset)
    ind_s, ind_e = 0, int(len(videos)*ratio)
    subset_labels = pickle.load(open(labels))[ind_s:ind_e]
    subset_videos = pickle.load(open(videos))[ind_s:ind_e]
    return subset_videos, subset_labels

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_label(label_path):
    f = h5py.File(data_root + '/' + label_path)
    labels = f['labels'].value
    behav, labels_count = np.unique(labels, return_counts=True)
    counts = {k:v for k,v in zip(behav,labels_count)}
    return list(labels), counts

def write_tfrecords(data_path,video_paths,action_labels,
                    n_vids_per_batch,subset,
                    n_frames_batch = 16,
                    n_frames_chunk = 512):
    """Function to write tfrecords.
        :param data_path: name of tfrecords file to write
        :param video_paths: list containing filenames of videos
        :param action_labels: list conatining filenames of ground truth label h5 files"""
    counts = {behav:0 for behav in L_POSSIBLE_BEHAVIORS}
    writer = tf.python_io.TFRecordWriter(data_path)
    video_count = 0
    tot_num_chunks = 0
    for i in tqdm(range(len(video_paths)),desc='Writing tf records..'):
        print '#'*80,'\n'
        video_name = video_paths[i].split('/')[-1]
        # print how many videos are saved every 1000 videos
        if (i!=0 and (not i % n_vids_per_batch)):
            print 'Train data: {}/{}\nVideo type:{}'.format(i, len(video_paths),type(video))
        # Load the video
        label, counts_curr = load_label(action_labels[i])
        for behav,count in counts_curr.iteritems():
            if behav.lower() != 'none':
                counts[behav] += count
        for ii in range(0, len(label),n_frames_chunk):
            j_range_max = min(len(label)-ii,n_frames_chunk) #load only as many frames for which labels are available
            video,(n,h,w,c) = load_video_with_path_cv2_abs("%s/%s" %(data_root,video_paths[i]),
                                                            starting_frame=ii,
                                                            n_frames=j_range_max)
            if type(video)==int:
                #Video does not exist, load video returned -1
                print "No video %s/%s exists %s"%(data_root,video_paths[i],video)
                continue
            if video.dtype != np.float32:
                video = video.astype(np.float32)
            #Incorporate shuffling within chunk
            curr_range = range(0,j_range_max-n_frames_batch)
            curr_num_chunks = len(curr_range)
            tot_num_chunks += curr_num_chunks
            shuffle(curr_range)
            for jj in tqdm(range(len(curr_range)),desc='Writing frames for chunk %s of video %s'%(ii/n_frames_chunk,video_name)):
                j = curr_range[jj] #Shuffled index j in current chunk
                vid = video[j:n_frames_batch+j]
                label_action = label[ii+n_frames_batch+j-1] #Add ii to account for starting frame number
                if label_action.lower() == 'none': #Do not train with 'none' labels that are present in the training h5 files
                    continue
                label_int = L_POSSIBLE_BEHAVIORS.index(label_action)
                # Create a feature
                feature = {'%s/label'%(subset): _int64_feature(label_int)}
                feature['%s/video'%(subset)] = _bytes_feature(tf.compat.as_bytes(vid.tostring()))
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                # Serialize to string and write on the file
                if example is not None:
                    writer.write(example.SerializeToString())
                    video_count += 1
                else:
        	    print "Example is None"
	            sys.stdout.flush()
    writer.close()
    sys.stdout.flush()
    return tot_num_chunks

def main():
    subset = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    videos, labels = get_lists(subset,0.3)
    print "Writing %s videos and labels"%(len(videos))
    tot_num_chunks = write_tfrecords('data/%s_0_3_flush_shuffled_norest_f32_mixed_mice.tfrecords'%(subset),videos, labels, 1, subset)
    print tot_num_chunks, "i chunks written"

if __name__=="__main__":
    main()
