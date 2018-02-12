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

def load_video_with_path_cv2(video_path, n_frames):
    """ Fuction to read a video, select a certain select number of frames, normalize and return the array of videos
    :param video_path: Path to the video that has to be loaded
    :param n_frames: Number of frames used to represent a video"""
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened()==False:
        return -1,-1
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ind_frames = map(int,np.linspace(0,video_length-1,n_frames))
    frameCount, index = 0,0
    vid = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frameCount += 1
            frame = cv2.resize(frame,(224,224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
            vid.append(frame)
        else:
            break
    curr_frames = [vid[i] for i in ind_frames]
    curr_frames = np.array(curr_frames)
    norm_frames = (curr_frames/127.5) - 1.
    return norm_frames,norm_frames.shape

def load_video_with_path_cv2_abs(video_path, starting_frame, n_frames):
    """ Fuction to read a video, convert all read frames into an array, normalize and return the array of videos
    :param video_path: Path to the video that has to be loaded
    :param n_frames: Number of frames used to represent a video"""
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened()==False:
        return -1,-1
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(1,starting_frame)
    frameCount, index = 0,0
    vid = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frameCount += 1
            frame = cv2.resize(frame,(224,224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
            vid.append(frame)
            if frameCount == n_frames:
                break
        else:
            break
    curr_frames = np.array(vid)
    norm_frames = (curr_frames/127.5) - 1.
    return norm_frames,norm_frames.shape


def print_preds_labels(preds,labels):
    """Function to print activity predictions and ground truth next to each other in words.
    :param preds: List of behavior predictions for a mini batch
    :param labels: List of behavior ground truth for the same mini batch"""
    for i,(prediction,ground_truth) in enumerate(zip(preds,labels)):
        print i,"Prediction: ",CLASSES_KIN[prediction],"Label: ",CLASSES_KIN[ground_truth]
    print list(preds==labels).count(True), "correct predictions"

def load_video_with_path(video_path, n_frames):
    """ Fuction to read a video, select a certain select number of frames, normalize and return the array of videos
    :param video_path: Path to the video that has to be loaded
    :param n_frames: Number of frames used to represent a video"""
    #imageio gives problems, seems unstable to read videos

    vid = imageio.get_reader(video_path,'ffmpeg')
    ind_frames = map(int,np.linspace(0,vid.get_length()-1,n_frames))
    curr_frames = [vid.get_data(i) for i in ind_frames]
    import ipdb; ipdb.set_trace()
    resized_frames = [cv2.resize(i,(224,224)) for i in curr_frames]
    curr_frames = np.array(resized_frames)
    #norm_frames = (curr_frames/127.5) - 1.
    return norm_frames,norm_frames.shape

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

def get_class_weights(LABELS_ROOT):
    import h5py
    import glob
    all_labels = glob.glob(LABELS_ROOT + '/*h5')
    for label in all_labels:
        f = h5py.File(label)
        labels = f['labels'].value
        label, count = np.unique(labels, return_counts=True)


def download_clip(video_identifier, output_dir,
                  start_time, end_time,
                  tmp_dir='/tmp/kinetics',
                  num_attempts=5,):
    """Trim video and store chunks.

    arguments:
    ---------
    video_identifier: str
        Path to the video stored on disk
    output_dir: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # Construct command to trim the videos (ffmpeg required).
    for i in range(0,3600):
        output_filename = output_dir + '/' + video_identifier + '_' + str(i) + '_' + str(i+1) + '.mp4'
        command = ['ffmpeg',
                   '-i', '"%s"' % tmp_filename,
                   '-ss', str(i),
                   '-t', '1',
                   '-c:v', 'libx264', '-c:a', 'copy',
                   '-threads', '1',
                   '-loglevel', 'panic',
                   '"%s"' % output_filename]
        command = ' '.join(command)
        try:
            output = subprocess.check_output(command, shell=True,
                                             stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            return status, err.output

        # Check if the video was successfully saved.
        status = os.path.exists(output_filename)
        os.remove(tmp_filename)
    return status, 'Downloaded'

def get_video_batch(video2label,batch_size=BATCH_SIZE,validation=False,val_ind=0,n_frames=N_FRAMES,class_index=True):
    """Function to return a random batch of videos for train mode and a specific set of videos for val mode.
        :param batch_size: Specifies the size of the batch of videos to be returned
        :param validation: Flag to specify training mode (True for val phase)
        :param val_ind: Index of the batch of val videos to retrieve"""

    if validation:
        VIDEOS_ROOT = KINETICS_ROOT + '/val'
        curr_videos = video2label.keys()[val_ind:val_ind+batch_size]
        curr_labels = [video2label[v] for v in curr_videos]
        #Implement videos that are missing.
        curr_video_paths = [glob.glob("%s/%s/%s*mp4"%(VIDEOS_ROOT,label,video_id))[0] for video_id,label in zip(curr_videos,curr_labels)]
        video_rgb_frames = [load_video_with_path_cv2(curr_vid,n_frames)[0] for curr_vid in curr_video_paths]
        #video_rgb_frames = [read_video(video_id,label,n_frames,'val')[0] for video_id, label in zip(curr_videos,curr_labels)]
        if class_index:
            curr_labels = [CLASSES_KIN.index(action) for action in curr_labels]
        return np.array(video_rgb_frames),np.array(curr_labels)
    else:
        VIDEOS_ROOT = KINETICS_ROOT + '/train'
        curr_inds = sample(0,len(video2label)-1,batch_size)
        curr_videos = video2label.keys()[curr_inds]
        curr_labels = [video2label[v] for v in curr_videos]
        curr_video_paths = [glob.glob("%s/%s/%s*mp4"%(VIDEOS_ROOT,label,video_id))[0] for video_id,label in zip(curr_videos,curr_labels)]
        video_rgb_frames = [load_video_with_path_cv2(curr_vid,n_frames)[0] for curr_vid in curr_video_paths]
        if class_index:
            curr_labels = [CLASSES_KIN.index(action) for action in curr_labels]
        return np.array(video_rgb_frames),np.array(curr_labels)
