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
# import cv2
from random import shuffle

# from google.colab import drive
# drive.mount('/content/gdrive')

def image_generator(files, batch_size=32, sz=(256, 256)):
    while True:

        # extract a random batch
        batch = np.random.choice(files, size=batch_size)

        # variables for collecting batches of inputs and outputs
        batch_x = []
        batch_y = []

        show = True
        for f in batch:
            # get the masks. Note tat masks are png files
            mask = Image.open(f.replace('vol', 'liver'))
            mask = np.array(mask.resize(sz))
            mask = mask / 255.

            batch_y.append(mask)

            # preprocess the raw images
            raw = Image.open(f'{f}')
            raw = raw.resize(sz)
            raw = np.array(raw)
            batch_x.append(raw)

        # preprocess a batch of images and masks
        batch_x = np.array(batch_x) / 255.
        batch_x = np.expand_dims(batch_x, 3)
        batch_y = np.array(batch_y)
        batch_y = np.expand_dims(batch_y, 3)

        yield (batch_x, batch_y)


batch_size = 1

all_files = []

for i in range(0, 131):
    all_files.append('gdrive/My Drive/EnlargedWithoutBorder_Dataset/enlarged_vol' + str(i) + 'a.png')
    all_files.append('gdrive/My Drive/EnlargedWithoutBorder_Dataset/enlarged_vol' + str(i) + 'c.png')
    all_files.append('gdrive/My Drive/EnlargedWithoutBorder_Dataset/enlarged_vol' + str(i) + 's.png')
    all_files.append('gdrive/My Drive/FlippedEnlargedWithoutBorder_Dataset/flipped_enlarged_vol' + str(i) + 'a.png')
    all_files.append('gdrive/My Drive/FlippedEnlargedWithoutBorder_Dataset/flipped_enlarged_vol' + str(i) + 'c.png')
    all_files.append('gdrive/My Drive/FlippedEnlargedWithoutBorder_Dataset/flipped_enlarged_vol' + str(i) + 's.png')
    for j in range(0, 360, 60):
        all_files.append(
            'gdrive/My Drive/RotatedEnlargedWithoutBorder_Dataset/rotated_' + str(j) + '_enlarged_vol' + str(
                i) + 'a.png')
        all_files.append(
            'gdrive/My Drive/RotatedEnlargedWithoutBorder_Dataset/rotated_' + str(j) + '_enlarged_vol' + str(
                i) + 'c.png')
        all_files.append(
            'gdrive/My Drive/RotatedEnlargedWithoutBorder_Dataset/rotated_' + str(j) + '_enlarged_vol' + str(
                i) + 's.png')

shuffle(all_files)

split = int(0.70 * len(all_files))

# split into training and testing
train_files = all_files[0:split]
test_files = all_files[split:]

train_generator = image_generator(train_files, batch_size=batch_size)
test_generator = image_generator(test_files, batch_size=batch_size)
x, y= next(train_generator)

plt.axis('off')
img = x[0].squeeze()
msk = y[0].squeeze()
img = np.stack((img,)*3, axis=-1)
msk = np.stack((msk,)*3, axis=-1)
plt.imshow( np.concatenate([img, msk, img*msk], axis = 1))

def dice_soft(y_true, y_pred, loss_type='sorensen', axis=[1,2,3], smooth=1e-5, from_logits=False):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.
    Parameters
    -----------
    y_pred : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    y_true : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    loss_type : string
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : list of integer
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
        If both y_pred and y_true are empty, it makes sure dice is 1.
        If either y_pred or y_true are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``,
        then if smooth is very small, dice close to 0 (even the image values lower than the threshold),
        so in this case, higher smooth can have a higher dice.
    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)
    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
    """

    if not from_logits:
        # transform back to logits
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
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice)
    return dice


def dice_hard(y_true, y_pred, threshold=0.5, axis=[1,2,3], smooth=1e-5):
    """Non-differentiable Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
    The coefficient between 0 to 1, 1 if totally match.
    Parameters
    -----------
    y_pred : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    y_true : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    threshold : float
        The threshold value to be true.
    axis : list of integer
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.
    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
    """
    y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    y_true = tf.cast(y_true > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(y_pred, y_true), axis=axis)
    l = tf.reduce_sum(y_pred, axis=axis)
    r = tf.reduce_sum(y_true, axis=axis)
    ## old axis=[0,1,2,3]
    # hard_dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # hard_dice = tf.clip_by_value(hard_dice, 0, 1.0-epsilon)
    ## new haodong
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice


def dice_loss(y_true, y_pred, from_logits=False):
    return 1-dice_soft(y_true, y_pred)

def specificity(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    tn = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 0), tf.equal(yp0, 0)))
    fp = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 0), tf.equal(yp0, 1)))
    specificity = tf.cast((tn)/(tn+fp), 'float32')
    return specificity
def precision(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    tn = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 0), tf.equal(yp0, 0)))
    fp = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 0), tf.equal(yp0, 1)))
    tp = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    fn = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 0)))
    precision = tf.where(tf.equal(tp+fp,0), 1., tf.cast(tp/(tp+fp), 'float32'))
    recall = tf.where(tf.equal(tp+fn,0), 1., tf.cast(tp/(tp+fn), 'float32'))
    f1 = tf.cast(2*precision*recall/(precision+recall), 'float32')
    return precision
def recall(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    tn = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 0), tf.equal(yp0, 0)))
    fp = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 0), tf.equal(yp0, 1)))
    tp = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    fn = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 0)))
    precision = tf.where(tf.equal(tp+fp,0), 1., tf.cast(tp/(tp+fp), 'float32'))
    recall = tf.where(tf.equal(tp+fn,0), 1., tf.cast(tp/(tp+fn), 'float32'))
    f1 = tf.cast(2*precision*recall/(precision+recall), 'float32')
    return recall
def f1(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    tn = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 0), tf.equal(yp0, 0)))
    fp = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 0), tf.equal(yp0, 1)))
    tp = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    fn = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 0)))
    precision = tf.where(tf.equal(tp+fp,0), 1., tf.cast(tp/(tp+fp), 'float32'))
    recall = tf.where(tf.equal(tp+fn,0), 1., tf.cast(tp/(tp+fn), 'float32'))
    f1 = tf.cast(2*precision*recall/(precision+recall), 'float32')
    return f1
def acc(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    tn = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 0), tf.equal(yp0, 0)))
    fp = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 0), tf.equal(yp0, 1)))
    tp = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    fn = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 0)))
    acc = tf.cast((tp+tn)/(tp+tn+fp+fn), 'float32')
    return acc
def voe(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.count_nonzero(yt0) + tf.count_nonzero(yp0)
    voe = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return 1-voe


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
    print("x is :", x)
    x = Conv2D(f, 3, activation=None, padding='same')(x)
    print("x is :", x)
    x = Conv2D(f, 3, activation=None, padding='same')(x)
    print("x is :", x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    print("Shivam outputs", outputs)
    print("Value of f :", f)

    # model creation
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=3e-5)
    model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics=[dice_hard, specificity, precision, recall, f1, acc, voe])

    return model
model = unet()
def main_predict():
    model.summary()
    model.load_weights('gdrive/My Drive/unet1-100.97final.h5')
    path = 'gdrive/My Drive/EnlargedWithoutBorder_Dataset/enlarged_vol4a.png'
    raw = Image.open(path)
    raw = np.array(raw.resize((256, 256))) / 255.
    raw = np.expand_dims(raw, 2)

    # predict the mask
    pred = model.predict(np.expand_dims(raw, 0))

    # mask post-processing
    raw = raw.squeeze()
    raw = np.stack((raw,) * 3, axis=-1)
    msk = pred.squeeze()
    msk = np.stack((msk,) * 3, axis=-1)
    print(np.unique(msk))
    msk[msk >= 0.5] = 1
    msk[msk < 0.5] = 0

    true_msk = Image.open(path.replace('vol', 'liver'))
    true_msk = np.array(true_msk.resize((256, 256))) / 255.
    true_msk = np.stack((true_msk,) * 3, axis=-1)

    # show the mask and the segmented image
    combined = np.concatenate([raw, msk, raw * msk, true_msk, raw * true_msk], axis=1)
    plt.axis('off')
    plt.imshow(msk)
    plt.show()

def build_callbacks():
        checkpointer = ModelCheckpoint(filepath='gdrive/My Drive/unetNewbce.h5', verbose=0, save_best_only=True,
                                       save_weights_only=True)
        callbacks = [checkpointer, PlotLearning()]
        return callbacks


# inheritance for training process plot
class PlotLearning(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        # self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('dice_hard'))
        self.val_acc.append(logs.get('val_dice_hard'))
        self.i += 1
        print('i=', self.i, 'loss=', logs.get('loss'), 'val_loss=', logs.get('val_loss'), 'dice_hard=',
              logs.get('dice_hard'), 'val_dice_hard=', logs.get('val_dice_hard'))

        # choose a random test image and preprocess
        path = np.random.choice(test_files)
        raw = Image.open(f'{path}')
        raw = np.array(raw.resize((256, 256))) / 255.
        raw = np.expand_dims(raw, 2)

        # predict the mask
        pred = model.predict(np.expand_dims(raw, 0))

        # mask post-processing
        raw = raw.squeeze()
        raw = np.stack((raw,) * 3, axis=-1)
        msk = pred.squeeze()
        msk = np.stack((msk,) * 3, axis=-1)
        print(np.unique(msk))
        msk[msk >= 0.5] = 1
        msk[msk < 0.5] = 0

        true_msk = Image.open(path.replace('vol', 'liver'))
        true_msk = np.array(true_msk.resize((256, 256))) / 255.
        true_msk = np.stack((true_msk,) * 3, axis=-1)

        # show the mask and the segmented image
        combined = np.concatenate([raw, msk, raw * msk, true_msk, raw * true_msk], axis=1)
        plt.axis('off')
        plt.imshow(combined)
        plt.show()

def train():
    train_steps = len(train_files) //batch_size
    test_steps = len(test_files) //batch_size
    model.fit_generator(train_generator,
                        epochs = 100, steps_per_epoch = train_steps,validation_data = test_generator, validation_steps = test_steps,
                        callbacks = build_callbacks(), verbose = 0)
    #model.save("gdrive/My Drive/unet2.h5")

def metrics_calculation():
    model.metrics_names
    train_steps = len(train_files) // batch_size
    model.evaluate_generator(train_generator, steps=train_steps, verbose=1)
    test_steps = len(test_files) // batch_size
    model.evaluate_generator(test_generator, steps=test_steps, verbose=1)
    generator = image_generator(all_files, batch_size=batch_size)
    steps = len(all_files) // batch_size
    model.evaluate_generator(generator, steps=steps, verbose=1)


def save_liver(file, destination_folder):
  index = file.rfind('/')
  raw = Image.open(file)
  raw = np.array(raw.resize((256,256)))/255.
  raw = np.expand_dims(raw,2)
  pred = model.predict(np.expand_dims(raw,0))
  raw = raw.squeeze()
  raw = np.stack((raw,)*3, axis = -1)
  msk = pred.squeeze()
  msk = np.stack((msk,)*3, axis = -1)
  msk[msk >= 0.5] = 1
  msk[msk < 0.5] = 0
  plt.axis('off')
  plt.imsave('gdrive/My Drive/' + destination_folder + '/'+ file[index+1:].replace('vol','liver'), raw*msk)
for i in range(0,131):
  print(i)
  file = ('gdrive/My Drive/EnlargedWithoutBorder_Dataset/enlarged_vol'+str(i)+'a.png')
  save_liver(file, 'Unet Liver Results')
  file = ('gdrive/My Drive/EnlargedWithoutBorder_Dataset/enlarged_vol'+str(i)+'c.png')
  save_liver(file, 'Unet Liver Results')
  file = ('gdrive/My Drive/EnlargedWithoutBorder_Dataset/enlarged_vol'+str(i)+'s.png')
  save_liver(file, 'Unet Liver Results')
  file = ('gdrive/My Drive/FlippedEnlargedWithoutBorder_Dataset/flipped_enlarged_vol'+str(i)+'a.png')
  save_liver(file, 'Flipped Unet Liver Results')
  file = ('gdrive/My Drive/FlippedEnlargedWithoutBorder_Dataset/flipped_enlarged_vol'+str(i)+'c.png')
  save_liver(file, 'Flipped Unet Liver Results')
  file = ('gdrive/My Drive/FlippedEnlargedWithoutBorder_Dataset/flipped_enlarged_vol'+str(i)+'s.png')
  save_liver(file, 'Flipped Unet Liver Results')
  for j in range(0,360,60):
    file = ('gdrive/My Drive/RotatedEnlargedWithoutBorder_Dataset/rotated_'+str(j)+'_enlarged_vol'+str(i)+'a.png')
    save_liver(file, 'Rotated Unet Liver Results')
    file = ('gdrive/My Drive/RotatedEnlargedWithoutBorder_Dataset/rotated_'+str(j)+'_enlarged_vol'+str(i)+'c.png')
    save_liver(file, 'Rotated Unet Liver Results')
    file = ('gdrive/My Drive/RotatedEnlargedWithoutBorder_Dataset/rotated_'+str(j)+'_enlarged_vol'+str(i)+'s.png')
    save_liver(file, 'Rotated Unet Liver Results')