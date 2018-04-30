import pickle
import glob
import tensorflow as tf
import numpy as np
import sys
import os
from tqdm import tqdm
from tf_utils import *
from fetch_balanced_batch import *

def compute_n_batch(H5_ROOT, batch_size):
    '''Function to compute number of samples
       to load for writing tfrecords.
       :param H5_ROOT: Root directory containing
                       all h5 files
       :param batch_size: Size of each minibatch'''
    all_h5_files = glob.glob('%s/*.h5'%(H5_ROOT))
    nLabels = 0
    for h5_f in h5_files:
        labels, counts = load_label(h5_f)
        nLabels += len(labels)
    return nLabels/batch_size

def write_tfrecords(VIDEO_ROOT,H5_ROOT,
                      batch_size=16,n_frames_batch = 16,
                      subset='train', n_frames_chunk = 512):
    """Function to write tfrecords.
        :param data_path: name of tfrecords file to write
        :param video_paths: list containing filenames of videos
        :param action_labels: list conatining filenames of
                              ground truth label h5 files"""

    counts = {behav:0 for behav in L_POSSIBLE_BEHAVIORS}
    writer = tf.python_io.TFRecordWriter(data_path)
    #Compute number of batches to write on tfrecords
    #to fully write our dataset
    n_batches = compute_n_batch(H5_ROOT, batch_size)
    #Dictionary mapping behavior to videos and
    #frame sequences in videos labeled as behavior
    behav2video = pickle.load(open('pickles/Behavior2Video.p'))
    ########## Start writing tfrecords ##########
    for i in tqdm(range(n_batches),
                    desc='Writing tf records..'):
        # Load the video
        label, counts_curr = load_label(action_labels[i])
        for behav,count in counts_curr.iteritems():
            if behav.lower() != 'none':
                counts[behav] += count

        ############### Read batches of video ###############

        for ii in tqdm(range(0, len(label),
                              n_frames_chunk),
                              desc='Reading batches of videos'):
            #load only as many frames for which labels are available
            j_range_max = min(len(label)-ii,n_frames_chunk)
            video,(n,h,w,c) = load_video_with_path_cv2_abs(
                                                    '%s/%s'%(
                                                    data_root,
                                                    video_paths[i],
                                                    dtype='uint8'),
                                                    starting_frame=ii,
                                                    n_frames=j_range_max)
            if type(video)==int:
                #Video does not exist, load video returned -1
                print "No video %s/%s exists %s"%(
                                                  data_root,
                                                  video_paths[i],
                                                  video
                                                  )
                continue
            if video.dtype != np.float32:
                video = video.astype(np.float32)
            #Incorporate shuffling within chunk
            curr_range = range(0,j_range_max-n_frames_batch)
            curr_num_chunks = len(curr_range)
            tot_num_chunks += curr_num_chunks
            shuffle(curr_range)
            for jj in tqdm(range(len(curr_range)),
                            desc='Writing frames for chunk %s of video %s'%(
                                                                ii/n_frames_chunk,
                                                                video_name
                                                                )):
                #Shuffled index j in current chunk
                j = curr_range[jj]
                vid = video[j:n_frames_batch+j]
                #Add ii to account for starting frame number
                label_action = label[ii+n_frames_batch+j-1]
                #Do not train with 'none' labels that are
                #present in the training h5 files
                if label_action.lower() == 'none':
                    continue
                label_int = L_POSSIBLE_BEHAVIORS.index(label_action)
                # Create a feature
                feature = {'%s/label'%(subset): _int64_feature(label_int)}
                feature['%s/video'%(subset)] = _bytes_feature(
                                                              tf.compat.as_bytes(
                                                              vid.tostring()
                                                              )
                                                              )
                # Create an example protocol buffer
                example = tf.train.Example(
                                        features=tf.train.Features(
                                        feature=feature
                                        ))
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
