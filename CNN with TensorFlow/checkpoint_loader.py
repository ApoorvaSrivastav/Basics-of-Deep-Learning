#from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

num_classes=10
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def create_model():
    model = Sequential()      #defining the model to be sequential
    #model structure can be seen in model.summary() in the cell below
    #relu is the activation function
    #same padding ensures same size
    #Dropout layers prevent overfitting
    #Conv2D layer provides 2D convolution
    #Dense layer deploys fully connected layers
    model.add(Conv2D(64, (3, 3), padding='same', 
              input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return(model)

import tensorflow as tf
#get the latest checkpoint file
checkpoint_path = "./train_ckpt/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)
latest = tf.train.latest_checkpoint(checkpoint_path)
print(latest)
# We create a new model, load the weights from the latest checkpoint and make inferences
# Create a new model instance
model_latest_checkpoint = create_model()
# Load the previously saved weights
model_latest_checkpoint.load_weights(latest)
# Re-evaluate the model
loss, acc = model_latest_checkpoint.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))  
