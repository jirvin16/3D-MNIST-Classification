from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf
import sys

from network import ConvNN

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 128, "Number of examples in minibatch [128]")
flags.DEFINE_integer("random_seed", 42, "Value of random seed [42]")
flags.DEFINE_integer("epochs", 80, "Number of epochs to run [80]")
flags.DEFINE_integer("hidden_dim", 128, "Size of hidden dimension [128]")
flags.DEFINE_integer("num_layers", 1, "Number of convolutional layers [1]")
flags.DEFINE_float("grad_max_norm", 5., "gradient max norm [1]")
flags.DEFINE_float("dropout", 0.2, "Dropout [0.2]")
flags.DEFINE_integer("mode", 0, "0 for training, 1 for testing, 2 for visualizing [0]")
flags.DEFINE_boolean("validate", True, "True for cross validation, False otherwise [True]")
flags.DEFINE_integer("save_every", 10, "Save every [10] epochs")
flags.DEFINE_string("model_name", "out", "model name for prefix to checkpoint file [unnamed]")
flags.DEFINE_boolean("convolution", True, "True if use convolution, False if regular network [True]")
flags.DEFINE_integer("out_channels", 32, "Number of out channels in filter [32]")
flags.DEFINE_string("optimizer", "Adam", "Gradient descent optimizer [Adam]")
flags.DEFINE_integer("voxel_dim", 16, "Number of voxels in a dimension (8, 16, 32) [16]")
flags.DEFINE_string("hidden_nonlinearity", "sigmoid", "Nonlinear activation after first fully connected layer [sigmoid]")

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)


def main(_):

    with tf.Session() as sess:
        attn = ConvNN(FLAGS, sess)
        attn.build_model()
        attn.run()


if __name__ == "__main__":
    tf.app.run()





