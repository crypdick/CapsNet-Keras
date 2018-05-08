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


model = load_model('result/trained_model.h5')

#model.load_weights('my_model_weights.h5')