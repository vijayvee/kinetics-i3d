import numpy as np
from scipy.misc import imread
import glob
import sys
import h5py
import pickle
from tqdm import tqdm

'''Create a mapping from video filename
    to behavior to frames labeled as
    the corresponding behavior'''

L_POSSIBLE_BEHAVIORS = ["drink",
                        "eat",
                        "groom",
                        "hang",
                        "sniff",
                        "rear",
                        "rest",
                        "walk",
                        "eathand"]

def load_label(label_path):
    f = h5py.File(label_path)
    labels = f['labels'].value
    behav, labels_count = np.unique(labels, return_counts=True)
    counts = {k:v for k,v in zip(behav,labels_count)}
    return list(labels), counts

def init_behav2video():
    behav2video = {b:{} for b in L_POSSIBLE_BEHAVIORS}
    return behav2video

def init_video2behav():
    video_files = glob.glob('%s/*'%(VIDEO_ROOT))
    video_files = [i.split('/')[-1] for i in video_files]
    video2behav = {
                    video_file:
                    {
                      b:[] for b in L_POSSIBLE_BEHAVIORS
                    } for video_file in video_files
                  }
    return video2behav

def populate_dict(h5_files, video_files, video2behav):
    for video_file, h5_file in tqdm(zip(video_files, h5_files),
                                      total=len(video_files),
                                      desc='Populating video2behav...'):
        video_fn = video_file.split('/')[-1]
        labels, counts = load_label(h5_file)
        for ind, label in enumerate(labels): #ind for frame number of activity
            if label.lower() != 'none':
                video2behav[video_fn][label] += [ind]
    return video2behav

def populate_dict_b2v(h5_files, video_files, behav2video, n_frames):
    for video_file, h5_file in tqdm(zip(video_files, h5_files),
                                      total=len(video_files),
                                      desc='Populating behav2video...'):
        video_fn = video_file.split('/')[-1]
        labels, counts = load_label(h5_file)
        for ind, label in enumerate(labels): #ind for frame number of activity
            if ind-n_frames < 0:
                continue
            if label.lower() != 'none':
                if behav2video[label].has_key(video_fn):
                    behav2video[label][video_fn] += [ind]
                else:
                    behav2video[label][video_fn] = [ind]
    return behav2video

def main():
    H5_ROOT = sys.argv[1]
    VIDEO_ROOT = sys.argv[2]
    BEHAV2VIDEO = sys.argv[3]
    DATASET_NAME = sys.argv[4]
    #all_h5 = glob.glob('%s/*.h5'%(H5_ROOT))
    #all_videos = glob.glob('%s/*.mp4'%(VIDEO_ROOT))
    #Load only small.pvd_40 videos, they have good labels
    all_h5 = pickle.load(open(
                    'pickles/small.pvd_40_gt_filenames.p'))
    all_videos = pickle.load(
                         open('pickles/small.pvd_40_video_filenames.p'))
    all_h5.sort()
    all_videos.sort()
    assert len(all_h5) == len(all_videos)
    if BEHAV2VIDEO == 'n': #y/n, y for behav2video
        video2behav = init_video2behav()
        video2behav = populate_dict(all_h5,
                                      all_videos,
                                      video2behav)
        pickle.dump(video2behav, open('Video2Behavior.p','w'))
    else:
        behav2video = init_behav2video()
        behav2video = populate_dict_b2v(all_h5,
                                          all_videos,
                                          behav2video,
                                          n_frames=16)
        pickle.dump(behav2video,
                     open('pickles/Behavior2Video_%s.p'%(
                                                    DATASET_NAME),
                                                    'w'))

if __name__=='__main__':
    main()
