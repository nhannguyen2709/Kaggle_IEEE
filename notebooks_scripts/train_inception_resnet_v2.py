# -*- coding: utf-8 -*-
"""
This script is used to fine-tune the InceptionResNetV2. Training process can be split into multiple stages.
"""
import tensorflow as tf
import keras.backend as K
from keras.utils import multi_gpu_model

from utils import train_validation_generator, load_pretrained_weights, freeze_layers, train_on_multi_gpus

K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


# arguments to set
NUM_GPUS = 4
IMG_HEIGHT_WIDTH = (299, 299)
INPUT_SHAPE = (299, 299, 3)
BATCH_SIZE = 64
OPTIMIZER = 'adam'
FREEZE_PROPORTION = 0.99
LIST_OF_NAMES = False

# data generators 
weights_path = "/home/nhannguyen/Kaggle_IEEE/model_weights/inception_resnet_v2/"
train_generator, validation_generator = train_validation_generator(batch_size=BATCH_SIZE,
                                                                   img_height_width=IMG_HEIGHT_WIDTH)
NUM_TRAIN_SAMPLES = 147060
NUM_VALID_SAMPLES = 1440
 
# create computational graph and load pre-trained ImageNet weights
inceptionresnetv2 = load_pretrained_weights('InceptionResNetV2', input_shape=INPUT_SHAPE)

# overwrite ImageNet weights with weights obtained from last training epoch
inceptionresnetv2.load_weights(weights_path + 'multigpu7.hdf5')

# fine-tune
freeze_layers(list_of_names=LIST_OF_NAMES,
              trainable_layers_names=None,
              model=inceptionresnetv2,
              freeze_proportion=FREEZE_PROPORTION)
parallel_model = multi_gpu_model(model=inceptionresnetv2, gpus=NUM_GPUS)
multi_stages_epochs = [10, 10, 10, 20, 20, 20, 20, 20]
multi_stages_learning_rate = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]
for i, stage_epochs in enumerate(multi_stages_epochs):
    history = train_on_multi_gpus(parallel_model=parallel_model,
                                  train_generator=train_generator,
                                  validation_generator=validation_generator,
                                  num_train_samples=NUM_TRAIN_SAMPLES, num_valid_samples=NUM_VALID_SAMPLES,
                                  batch_size=BATCH_SIZE, optimizer=OPTIMIZER, 
                                  learning_rate=multi_stages_learning_rate[i], 
                                  epochs=stage_epochs)
    save_path = weights_path + 'multigpu' + str(i) + '.hdf5'
    inceptionresnetv2.save_weights(save_path)
    print("Successfully saved weights obtain from training stage {} onto disk".format(i))