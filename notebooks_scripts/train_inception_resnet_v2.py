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

weights_path = "/home/nhannguyen/Kaggle_IEEE/model_weights/inception_resnet_v2/"

train_generator, validation_generator = train_validation_generator(batch_size=50,
                                                                   img_height_width=(299, 299))
 
# create computational graph and load pre-trained ImageNet weights
inceptionresnetv2 = load_pretrained_weights('InceptionResNetV2', input_shape=(299, 299, 3))

# overwrite ImageNet weights with weights obtained from last training epoch
inceptionresnetv2.load_weights(weights_path + 'multigpu.hdf5')

# fine-tune
parallel_model = multi_gpu_model(model=inceptionresnetv2, gpus=3)
freeze_layers(list_of_names=False,
              trainable_layers_names=None,
              model=inceptionresnetv2,
              freeze_proportion=0.9)
multi_stages_epochs = [5, 5, 10, 10, 10, 20, 20, 20]
for i, stage_epochs in enumerate(multi_stages_epochs):
    history = train_on_multi_gpus(parallel_model=parallel_model,
                                  train_generator=train_generator,
                                  validation_generator=validation_generator,
                                  num_train_samples=490050, num_valid_samples=4950,
                                  batch_size=50, optimizer='adam', 
                                  learning_rate=1e-3, epochs=stage_epochs)
    save_path = weights_path + 'multigpu' + str(i) + '.hdf5'
    inceptionresnetv2.save_weights(save_path)
    print("Successfully saved weights obtain from training stage {} onto disk".format(i))