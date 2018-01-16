import tensorflow as tf
import keras.backend as K
from keras.utils import multi_gpu_model

from utils import train_validation_generator, load_pretrained_weights, freeze_layers, train_on_multi_gpus

K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

weights_path = "/home/nhannguyen/sigprocsoccomp/model_weights/inception_resnet_v2/"

train_generator = train_validation_generator(batch_size=36,
                                             img_height_width=(512, 512))[0]
validation_generator = train_validation_generator(batch_size=36,
                                                  img_height_width=(512, 512))[1]
 
# create computational graph and load pre-trained ImageNet weights
inceptionresnetv2 = load_pretrained_weights('InceptionResNetV2', input_shape=(512, 512, 3))

# overwrite ImageNet weights with weights obtained from last training epoch
inceptionresnetv2.load_weights(weights_path + 'multigpu.hdf5')

# fine-tune
parallel_model = multi_gpu_model(model=inceptionresnetv2, gpus=4)

freeze_layers(list_of_names=False,
              trainable_layers_names=None,
              model=inceptionresnetv2,
              freeze_proportion=0.9)

history = train_on_multi_gpus(parallel_model=parallel_model,
                              train_generator=train_generator,
                              validation_generator=validation_generator,
                              num_train_samples=22319, num_valid_samples=2430,
                              batch_size=36,
                              optimizer='adam', learning_rate=1e-4, epochs=200)

freeze_layers(list_of_names=True,
              trainable_layers_names=['dense_0' ,'dense_1', 'new_predictions'],
              model=inceptionresnetv2,
              freeze_proportion=0)

history = train_on_multi_gpus(parallel_model=parallel_model,
                              train_generator=train_generator,
                              validation_generator=validation_generator,
                              num_train_samples=22319, num_valid_samples=2430,
                              batch_size=36,
                              optimizer='sgd', learning_rate=5e-3, epochs=10)

# save weights after finishing training
inceptionresnetv2.save_weights(weights_path + 'multigpu.hdf5')
print("Successfully saved weights onto disk")