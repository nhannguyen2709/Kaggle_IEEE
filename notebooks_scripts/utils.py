# -*- coding: utf-8 -*-
"""
utility functions
"""
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
import keras.backend as K
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils import multi_gpu_model
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

from inception_resnet_v2 import InceptionResNetV2

K.set_image_dim_ordering('tf')

train_dir = "/mnt/safe01/data/train/"
validation_dir = "/mnt/safe01/data/validation/"
test_dir = "/mnt/safe01/data/tests/"

def train_validation_generator(batch_size, img_height_width):
    """
    Return train and validation data generator.
    """
    # augmentation configuration for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)
    # augmentation configuration for testing: only rescaling
    validation_datagen = ImageDataGenerator(
        rescale=1. / 255)    
    # function to generate batches for training the network
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=batch_size,
        target_size=img_height_width,
        class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        batch_size=batch_size,
        target_size=img_height_width,
        class_mode='categorical')
    return train_generator, validation_generator

# load pre-trained weights onto CPU
def load_pretrained_weights(network_name, input_shape):
    if network_name == 'InceptionResNetV2':
        with tf.device('/cpu:0'):
            model = InceptionResNetV2(include_top=False, pooling='avg_max',
                                      input_shape=input_shape)
    elif network_name == 'ResNet50':
        with tf.device('/cpu:0'):
            model = ResNet50(include_top=False, input_shape=input_shape)
    elif network_name == 'ResNet152':
        with tf.device('/cpu:0'):
            model = ResNet152(img_rows=512, img_cols=512, num_channels=3, num_classes=10)
    return model

# freeze layers
def freeze_layers(list_of_names, model, trainable_layers_names, freeze_proportion=0):
    """
    If list_of_names is True then we can specify which layers to be freezed by trainable_layers_names. Another way is to specify the number of layers by freeze_proportion eg freeze_proportion=0.5 means half of the top layers are freezed when training.
    """
    if list_of_names == True:
        trainable_layers = [model.get_layer(layer_name)
                            for layer_name in trainable_layers_names]
        for layer in model.layers:
            if layer not in trainable_layers:
                layer.trainable = False
    else:
        num_layers = len(model.layers)
        for layer in model.layers[:int(num_layers*freeze_proportion)]:
            layer.trainable = False
        for layer in model.layers[int(num_layers*freeze_proportion):]:
            layer.trainable = True
            
def train_on_multi_gpus(parallel_model, train_generator, 
                        validation_generator, num_train_samples, num_valid_samples,
                        batch_size, optimizer, learning_rate, epochs):
    """
    Wrap the fit_generator() function of a parallel model inside and return the History object.
    """
    if optimizer == 'sgd':
        parallel_model.compile(loss='categorical_crossentropy',
                               optimizer=SGD(lr=learning_rate,
                                             decay=learning_rate / epochs),
                               metrics=['accuracy'])
        history = parallel_model.fit_generator(
            train_generator,
            steps_per_epoch=num_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=num_valid_samples // batch_size)
    elif optimizer == 'adam':
        parallel_model.compile(loss='categorical_crossentropy',
                               optimizer=Adam(lr=learning_rate),
                               metrics=['accuracy'])
        history = parallel_model.fit_generator(
            train_generator,
            steps_per_epoch=num_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=num_valid_samples // batch_size)
    elif optimizer == 'rmsprop':
        parallel_model.compile(loss='categorical_crossentropy',
                               optimizer=RMSprop(lr=learning_rate, decay=0.9),
                               metrics=['accuracy'])
        history = parallel_model.fit_generator(
            train_generator,
            steps_per_epoch=num_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=num_valid_samples // batch_size)
    return history