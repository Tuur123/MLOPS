import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

import keras

from keras import backend as K
from keras import regularizers
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Masking, Activation

import matplotlib.pyplot as plt

def custom_loss(y_true,y_pred):
    y_mask=keras.backend.clip(y_true, 0, 0.01)*100
    return K.mean(K.square(y_mask*(y_pred - y_true)), axis=-1)


def load_data(path):
    X_train = np.load(path + "/X_train.npy", allow_pickle=True)
    y_train = np.load(path + "/y_train.npy", allow_pickle=True)
    X_test = np.load(path + "/X_test.npy", allow_pickle=True)
    y_test = np.load(path + "/y_test.npy", allow_pickle=True)
    ratedlist = np.load(path + "/ratedlist.npy", allow_pickle=True)
    return X_train, y_train, X_test, y_test, ratedlist


def train_model(X_train, y_train, X_test, y_test, batchsize, epochs, optimizer):
    input_img = Input(shape=(X_train.shape[1],)) # All movies

    encoded = Masking(mask_value=0)(input_img)

    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(50, activation='relu')(encoded)
    encoded = Dense(12, activation='relu')(encoded)

    decoded = Dense(50, activation='relu')(encoded)
    decoded = Dense(12, activation='relu')(decoded)
    decoded = Dense(y_train.shape[1], activation='sigmoid')(decoded) # All movies

    autoencoder = Model(input_img, decoded)

    autoencoder.compile(loss=custom_loss, optimizer=optimizer)

    split = int(X_train.shape[0] * 0.8)

    autoencoder.fit(X_train, y_train, epochs=epochs, batch_size=batchsize, validation_data=(X_test, y_test))

    autoencoder.save("../snapshots/imdb-model.h5")



