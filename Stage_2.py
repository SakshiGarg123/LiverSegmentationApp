import keras
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import cv2
import logging
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
from random import shuffle
from PIL import Image
from sklearn.model_selection import train_test_split
from keras import applications

def get_model():
    input_tensor = Input(shape=(256, 256, 3))
    top_model_weights_path = 'weights/bottleneck_fc_model.h5'
    model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    print('Model loaded')

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    top_model.load_weights(top_model_weights_path)

    new_model = Sequential()
    new_model.add(model)
    new_model.add(top_model)

    for layer in new_model.layers[:25]:
        layer.trainable = False

    new_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=3e-5), metrics=['accuracy'])
    new_model.load_weights('weights/transfer_learning_weights.h5')
    return new_model

def main_predict(model, path):
    raw = Image.open(path)
    raw = np.array(raw.resize((256, 256))) / 255.
    raw = np.stack((raw,) * 3, axis=-1)
    # predict the mask
    pred = model.predict(np.expand_dims(raw, 0))
    print(pred[0][0])
    return pred[0][0]

model = get_model()
main_predict(model, 'C:/Users/Soumya/Desktop/Cropped Dataset/Cropped Dataset/enlarged_liver4a.png')