import keras                               #importing keras library
from keras.datasets import cifar10         #importing dataset cifar10 from keras dataset
from keras.models import Sequential        #importing model for sequential neural network structure
                                           #which allows addition of layers in the network 
from keras.layers import Dense, Dropout, Activation, Flatten #keras has multiple definitions of
from keras.layers import Conv2D, MaxPooling2D                #standard layers to be used while 
                                                             #constructing a ntwork they are called as a function
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import History

batch_size = 512           #generally chosen in terms of 32,64,128.
                          #Represents the number of samples after which weights are updated
num_classes = 10          #the categories in which the data has to be classified
epochs = 50              #Epochs is the number of times whole training data is passed to the deep net
save_dir = os.path.join(os.getcwd(), 'saved_models') #saving the model in specified folder
model_name = 'model3.h5'        #final name of the model
 
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
# Doing this only one bit will be high out of 10 bits in the output of 10x1 vector
y_train= keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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
model.add(Dropout(0.1))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

checkpoint_path = "train_ckpt/cp.ckpt"
# Create a callback that saves the model's weights every 10 epochs as checkpoints
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=0, 
    save_weights_only=False,
    period=10)

# initiate Adam optimizerlearning rate 0.001
opt=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# finalizing the model by defining loss,optimization and the evaluation metric
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#training the model
history=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,callbacks=[cp_callback])


# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
