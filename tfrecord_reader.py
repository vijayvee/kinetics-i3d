#!/usr/bin/python
"""File to read tfrecords and return labels,videos"""
import tensorflow as tf
import numpy as np
import sys
import os

def get_video_label_tfrecords(filename_queue,batch_size,subset,shuffle=False):
    feature = {'{}/video'.format(subset): tf.FixedLenFeature([], tf.string),
              '{}/label'.format(subset): tf.FixedLenFeature([], tf.int64)}
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Cast label data into int32
    label = tf.cast(features['{}/label'.format(subset)], tf.int64)
    video_dec = tf.decode_raw(features['{}/video'.format(subset)],np.float32)
    # Reshape image data into the original shape
    video = tf.reshape(video_dec, [16, 224, 224, 3])
    # Creates batches by randomly shuffling tensors
    if shuffle:
        videos, labels = tf.train.shuffle_batch([video, label], batch_size=batch_size, capacity=30, num_threads=100, min_after_dequeue=10)
        return videos,labels
    videos, labels = tf.train.batch([video, label], batch_size=batch_size, capacity=30, num_threads=100)
    return videos, labels

def test_tfrecord_read(tfrecords_filename):
    with tf.Session().as_default() as sess:
        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=1)
        videos,labels = get_video_label_tfrecords(filename_queue,64,subset='train')
        init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        videos_batch,labels_batch = sess.run([videos,labels])
        print videos_batch.shape,labels_batch
def main():
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    test_tfrecord_read(sys.argv[1])

if __name__=="__main__":
    main()
