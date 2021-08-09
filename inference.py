#! /usr/bin/python
# -*- coding: utf8 -*-

""" GAN-CLS """
import tensorflow.compat.v1 as tf
import tensorlayer as tl

from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
from scipy.io import loadmat
import time, os, re, nltk
import pickle
from utils import *
from model import *
import model




save_dir = "checkpoint"
batch_size = 64
z_dim = 512         # Noise dimension
image_size = 64     # 64 x 64
c_dim = 3           # for rgb
ni = int(np.ceil(np.sqrt(batch_size)))

tf.disable_eager_execution()

with open("_vocab.pickle", 'rb') as f:
    vocab = pickle.load(f)

t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name='real_image')

t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')

t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

generator_txt2img = model.generator_txt2img_resnet

net_rnn = model.rnn_embed(t_real_caption, is_train=False, reuse=False)
net_g, _ = generator_txt2img(t_z,
                             net_rnn.outputs,
                             is_train=False, reuse=False, batch_size=batch_size)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
tl.layers.initialize_global_variables(sess)

net_rnn_name = os.path.join(save_dir, 'net_rnn.npz')
net_cnn_name = os.path.join(save_dir, 'net_cnn.npz')
net_g_name = os.path.join(save_dir, 'net_g.npz')
net_d_name = os.path.join(save_dir, 'net_d.npz')

net_rnn_res = tl.files.load_and_assign_npz(sess=sess, name=net_rnn_name, network=net_rnn)
net_g_res = tl.files.load_and_assign_npz(sess=sess, name=net_g_name, network=net_g)

sample_size = batch_size


sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
        # sample_seed = np.random.uniform(low=-1, high=1, size=(sample_size, z_dim)).astype(np.float32)]
n = int(sample_size / ni)


sample_sentence = ["A man walking a dog."] * n + \
                      ["Two dogs."] * n + \
                      ["A small black dog."] * n + \
                      ["A dog playing with frisbee."] * n + \
                      ["A dog sitting. "] * n + \
                      ["A woman and dog."] * n + \
                      ["Brown dog sitting."] * n +\
                      ["A dog laying on the grass."] * n


for i, sentence in enumerate(sample_sentence):
    sentence = preprocess_caption(sentence)
    sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [
        vocab.end_id]  # add END_ID
    sample_sentence[i] = [vocab.word_to_id(word) for word in sentence]

sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')
img_gen, rnn_out = sess.run([net_g.outputs, net_rnn_res.outputs], feed_dict={
                                        t_real_caption : sample_sentence,
                                        t_z : sample_seed})
save_images(img_gen, [ni, ni], 'generated_image.png')
