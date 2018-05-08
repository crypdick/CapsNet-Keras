from keras.callbacks import ModelCheckpoint
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from six.moves import cPickle as pickle
from six.moves import range
from sklearn.model_selection import train_test_split
from keras.models import load_model
from capsulenet_vanilla_mnist import load_mnist, load_notMNIST_from_npy, CapsNet
import tensorflow as tf


import os
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

# setting the hyper parameters
parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.001, type=float,
                    help="Initial learning rate")
parser.add_argument('--lr_decay', default=0.9, type=float,
                    help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
parser.add_argument('--lam_recon', default=0.392, type=float,
                    help="The coefficient for the loss of decoder")
parser.add_argument('-r', '--routings', default=3, type=int,
                    help="Number of iterations used in routing algorithm. should > 0")
parser.add_argument('--shift_fraction', default=0.1, type=float,
                    help="Fraction of pixels to shift at most in each direction.")
parser.add_argument('--debug', action='store_true',
                    help="Save weights by TensorBoard")
parser.add_argument('--save_dir', default='./result')
parser.add_argument('-t', '--testing', action='store_true',
                    help="Test the trained model on testing dataset")
parser.add_argument('--digit', default=5, type=int,
                    help="Digit to manipulate")
parser.add_argument('-w', '--weights', default=None,
                    help="The path of the saved weights. Should be specified when testing")
args = parser.parse_args()
print(args)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# load data
(x_train, y_train), (x_test, y_test) = load_mnist()
x_train1, y_train1, x_test1, y_test1 = load_notMNIST_from_npy()

# define model
model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                              n_class=len(np.unique(np.argmax(y_train, 1))),
                                              routings=args.routings)
model.summary()

# train or test
#model.load_weights(args.weights)
model.load_weights('mnist_vanilla_result/weights-04.h5')
saver = tf.train.Saver()
saver.save(K.get_session(), '/tmp/keras_model.ckpt')
