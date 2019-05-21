import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose
from keras.layers import Concatenate, BatchNormalization, UpSampling2D
from keras.layers import  Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.utils import plot_model
import tensorflow as tf
import glob
import random
from random import shuffle

def dice_soft(y_true, y_pred, loss_type='sorensen', axis=[1,2,3], smooth=1e-5, from_logits=False):
    if not from_logits:
        _epsilon = tf.convert_to_tensor(1e-7, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
        y_pred = tf.log(y_pred / (1 - y_pred))

    inse = tf.reduce_sum(y_pred * y_true, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(y_pred * y_pred, axis=axis)
        r = tf.reduce_sum(y_true * y_true, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(y_pred, axis=axis)
        r = tf.reduce_sum(y_true, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice

def dice_hard(y_true, y_pred, threshold=0.5, axis=[1,2,3], smooth=1e-5):
    y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    y_true = tf.cast(y_true > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(y_pred, y_true), axis=axis)
    l = tf.reduce_sum(y_pred, axis=axis)
    r = tf.reduce_sum(y_true, axis=axis)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice

def dice_loss(y_true, y_pred, from_logits=False):
    return 1-dice_soft(y_true, y_pred)

def unet(sz=(256, 256, 1)):
    x = Input(sz)
    inputs = x
    # down sampling
    f = 8
    layers = []
    for i in range(0, 6):
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        layers.append(x)
        x = MaxPooling2D()(x)
        f = f * 2
    ff2 = 64
    # bottleneck
    j = len(layers) - 1
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j - 1
    # upsampling
    for i in range(0, 5):
        ff2 = ff2 // 2
        f = f // 2
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=3)([x, layers[j]])
        j = j - 1
        # classification
    x = Conv2D(f, 3, activation=None, padding='same')(x)
    x = Conv2D(f, 3, activation=None, padding='same')(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    # model creation
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=3e-5)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[dice_hard])

    return model
def get_model():
    model = unet()
    model.load_weights('weights/unet1-100.97final.h5')
    return model
def main_predict(model, path):
    raw = Image.open(path)
    raw = np.array(raw.resize((256, 256))) / 255.
    raw = np.expand_dims(raw, 2)
    # predict the mask
    pred = model.predict(np.expand_dims(raw, 0))
    msk = pred.squeeze()
    msk = np.stack((msk,) * 3, axis=-1)
    msk[msk >= 0.5] = 1
    msk[msk < 0.5] = 0
    return msk

