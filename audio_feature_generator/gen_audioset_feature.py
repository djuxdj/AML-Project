from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import os
import tqdm

flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'audioset/vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS

audio_path = '/home/audio-visual/WavFiles/test/'
dest_path = '/home/audio-visual/data/MSR-VTT/features/test/Audio_features/'


with tf.device('/device:GPU:2'):
  def main(_):
    audio_files = os.listdir(audio_path)
    # maxi = 0
    for each_file in tqdm.tqdm(audio_files):
      wav_file = audio_path+each_file
      examples_batch = vggish_input.wavfile_to_examples(wav_file)
      # print(examples_batch.shape)
      
      with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        [embedding_batch] = sess.run([embedding_tensor],feed_dict={features_tensor: examples_batch})
        postprocessed_batch = embedding_batch
        np.save(dest_path+each_file.split('.')[0]+'.npy',postprocessed_batch)
        # print(postprocessed_batch.shape)
        # break
        # maxi = max(maxi,postprocessed_batch.shape[0])
        # print("I am --------------->"+str(maxi))
      # os.system('rm output.wav')
      # break


  if __name__ == '__main__':
    tf.app.run()
