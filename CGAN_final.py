# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:18:38 2021

@author: SDW
"""
# %% Importing libraries
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
from scipy.io import wavfile
import numpy as np
import array
import os, fnmatch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#%%  Importing Tensorflow and sklearn tools
import tensorflow as tf
print('tensorflow version: ', tf.__version__)
import keras
print('keras version: ',keras.__version__)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM,GaussianNoise,Bidirectional,TimeDistributed
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,ProgbarLogger
from keras.utils import np_utils
from keras import backend as K
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

