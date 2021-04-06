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
#%% A conditional gan 
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.models import Model
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.layers.normalization import BatchNormalization
#%% subject file loading
INPUT_DIR='sub_1'
Img_files = fnmatch.filter(os.listdir(INPUT_DIR), 'imgkp*')
#%%
def last_4chars(x):
    return(x[:-8])

srt=sorted(Img_files, key = last_4chars)   
#%% define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1), n_classes=3):
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)
    li = Reshape((in_shape[0], in_shape[1], 1))(li)
    in_image = Input(shape=in_shape)
    merge = Concatenate()([in_image, li])
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)
    out_layer = Dense(1, activation='sigmoid')(fe)
    model = Model([in_image, in_label], out_layer)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
#%% define the standalone generator model
def define_generator(latent_dim, n_classes=3):
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = 7 * 7
    li = Dense(n_nodes)(li)
    li = Reshape((7, 7, 1))(li)
    in_lat = Input(shape=(latent_dim,))
    n_nodes = 128 * 7 * 7
    gen = Dense(n_nodes)(in_lat)
    gen =BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 128))(gen)
    merge = Concatenate()([gen, li])
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
    gen =BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen =BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
    model = Model([in_lat, in_label], out_layer)
    return model 
#%% define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    d_model.trainable = False
    gen_noise, gen_label = g_model.input
    gen_output = g_model.output
    gan_output = d_model([gen_output, gen_label])
    model = Model([gen_noise, gen_label], gan_output)
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
#%% # select real samples
def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples, 1))
    return [X, labels], y
#%% generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=3):
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]
#%% use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict([z_input, labels_input])
    y = zeros((n_samples, 1))
    return [images, labels_input], y
#%% numpy array
from keras.preprocessing.image import  img_to_array
from PIL import Image
x=0
img_array=[]
for x in range(99):
    task=x+1
    image_name="imgkp28gr_"+str(task)+".png"
    print(image_name)
    
    img = Image.open(image_name) 
    X = img_to_array(img) 
    X=np.interp(X, (X.min(), X.max()), (-1, +1))
    img_array.append(X)
#%%

data_pd = pd.read_csv('sub1_arget.csv',header=None)
tar_val=np.asarray(data_pd )
tar_val=tar_val-1

seed = 145

x_train, x_test, y_train, y_test = train_test_split(img_array, tar_val ,train_size=0.7, random_state=seed, shuffle=True)
y_actual=y_test
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state=seed, shuffle=True)    
#%% load real samples
def load_real_samples():    
     X =np.asarray(x_train) 
     return [X, y_train]

#%% create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.plot(d1_hist, label='d-real')
    pyplot.plot(d2_hist, label='d-fake')
    pyplot.plot(g_hist, label='gen')
    pyplot.legend()
    pyplot.subplot(2, 1, 2)
    pyplot.plot(a1_hist, label='acc-real')
    pyplot.plot(a2_hist, label='acc-fake')
    pyplot.legend()
    pyplot.savefig('results_opt/plot_line_plot_loss.png')
    pyplot.close()
#%% train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1000, n_batch=8):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
            d_loss1,d_acc1 = d_model.train_on_batch([X_real, labels_real], y_real)
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2,d_acc2 = d_model.train_on_batch([X_fake, labels], y_fake)
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
               
            d1_hist.append(d_loss1)
            d2_hist.append(d_loss2)
            g_hist.append(g_loss)
            a1_hist.append(d_acc1)
            a2_hist.append(d_acc2)
        
    plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)
    g_model.save('cgan_generator.h5')
#%% 
latent_dim = 100
d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)
dataset = load_real_samples()
train(g_model, d_model, gan_model, dataset, latent_dim)
#%%
# example of loading the generator model and generating images
from numpy import asarray
from keras.models import load_model
from matplotlib import pyplot
 

def save_plot(examples, n):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i, :, :, 0], cmap='rainbow')
    pyplot.show()

g_model = load_model('cgan_generator.h5')
latent_points, labels = generate_latent_points(100, 9)
labels = asarray([x for _ in range(3) for x in range(3)])
X  = g_model.predict([latent_points, labels])
X = (X + 1) / 2.0
save_plot(X, 3)
