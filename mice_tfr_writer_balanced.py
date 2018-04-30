import pickle
import glob
import tensorflow as tf
import numpy as np
import sys
import os
from tqdm import tqdm
from tf_utils import *
from fetch_balanced_batch import *
import h5py

H5_ROOT = '/media/data_cifs/mice/mice_data_2018/labels'
DATASET_NAME = sys.argv[1]
SUBSET = sys.argv[2]
OUTPUT_PATH = sys.argv[3]

#TODO: Add strftime to the written tfrecord filename

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def compute_n_batch(H5_ROOT, batch_size):
    '''Function to compute number of samples
       to load for writing tfrecords.
       :param H5_ROOT: Root directory containing
                       all h5 files
       :param batch_size: Size of each minibatch'''
    all_h5_files = glob.glob('%s/*.h5'%(H5_ROOT))
    nLabels = 0
    for h5_f in all_h5_files:
        labels, counts = load_label(h5_f)
        nLabels += len(labels)
    return nLabels/batch_size

def write_tfrecords(subset='train',
                      n_batches=None,
                      batch_size=16):
    """Function to write tfrecords.
        :param subset: One of 'train' or 'va'
                       indicating the dataset
                       being written"""

    counts = {behav:0 for behav in L_POSSIBLE_BEHAVIORS}
    writer = tf.python_io.TFRecordWriter(OUTPUT_PATH)
    #Compute number of batches to write on tfrecords
    #to fully write our dataset
    n_batches = compute_n_batch(H5_ROOT, batch_size)

    #Dictionary mapping behavior to videos and
    #frame sequences in videos labeled as behavior
    b2v_pickle = 'pickles/Behavior2Video_%s_%s.p'%(
                                        DATASET_NAME,
                                        subset
                                        )
    behav2video = pickle.load(open(b2v_pickle))
    ########## Start writing tfrecords ##########
    for i in tqdm(range(n_batches),
                    desc='Writing tf records..'):
        # Load the video
        video_chunks, labels = fetch_balanced_batch(behav2video)
        for behav in labels:
            counts[behav] += 1
        #Convert labels to discrete category indices
        labels_int = [L_POSSIBLE_BEHAVIORS.index(label)
                          for label in labels]
        for i in tqdm(range(len(labels_int)),
                        desc='Writing single minibatch'):
            ########## Create tfrecord features ##########
            X = video_chunks[i,:,:,:,:]
            y = labels_int[i]
            feature = {'%s/label'%(subset):
                        _int64_feature(y)}
            feature['%s/video'%(subset)] = _bytes_feature(
                                                    tf.compat.as_bytes(
                                                    X.tostring())
                                                    )
            ########## Create an example protocol buffer #
            example = tf.train.Example(
                                    features=tf.train.Features(
                                    feature=feature
                                    ))
            ## Serialize to string and write on the file #
            if example is not None:
                writer.write(example.SerializeToString())
            else:
                print "Example is None"
        if i%500==0:
            sys.stdout.flush()
    writer.close()
    sys.stdout.flush()

def main():
    write_tfrecords(subset=SUBSET)

if __name__=='__main__':
    main()
