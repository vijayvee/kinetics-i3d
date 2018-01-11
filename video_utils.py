#!/usr/bin/python
import imageio
import numpy as np
import cv2
"""Script for video related utilities"""

def read_video(video_fn, n_frames):
    """Function to read a single video given by video_fn and return n_frames equally spaced frames from the video
        video_fn: Filename of the video to read
        n_frames: Number of frames to read in the video"""

    vid = imageio.get_reader(video_fn,'ffmpeg')
    ind_frames = map(int,np.linspace(0,vid.get_length()-1,n_frames))
    curr_frames = [vid.get_data(i) for i in ind_frames]
    resized_frames = [cv2.resize(i,(224,224)) for i in curr_frames]
    curr_frames = np.array(resized_frames)
    norm_frames = (curr_frames/127.5) - 1.
    return norm_frames,norm_frames.shape
