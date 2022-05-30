import tensorflow.keras
from tensorflow.keras.datasets import cifar10,mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from time import time
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,LSTM,Dropout
from tensorflow.keras.utils import normalize
# Model configuration
img_width, img_height = 32, 32
batch_size = 250
no_epochs = 25
no_classes = 10
validation_split = 0.2
verbosity = 1
# Load CIFAR10 dataset
# (input_train,target_train),(input_test,target_test) = mnist.load_data()
# input_train,input_test = normalize(input_train,axis = 1),normalize(input_test,axis = 1)
# (input_train, target_train), (input_test, target_test) = cifar10.load_data()
# Set input shape
# input_shape = (img_width, img_height, 3)
# input_shape = (img_width, img_height)

# # Parse numbers as floats
# input_train = input_train.astype('float32')
# input_test = input_test.astype('float32')
#
# # Normalize data
# input_train = input_train / 255
# input_test = input_test / 255

# Convert target vectors to categorical targets
# target_train = tensorflow.keras.utils.to_categorical(target_train, no_classes)
# target_test = tensorflow.keras.utils.to_categorical(target_test, no_classes)

# model = Sequential()
# model.add(LSTM(128,input_shape = (input_train.shape[1:]),activation = 'relu'))
# # model.add(LSTM(128,input_shape = input_shape,activation = 'relu',return_sequences = True))
# model.add(Dropout(0.2))
#
# model.add(LSTM(128,activation = 'relu'))
# model.add(Dropout(0.2))
#
# model.add(Dense(32,activation = 'relu'))
# model.add(Dropout(0.2))
#
# model.add(Dense(10,activation = 'softmax'))
#
#
# # Compile the model
# model.compile(loss="sparse_categorical_crossentropy",
#               optimizer=tensorflow.keras.optimizers.Adam(lr = 1e-3,decay = 1e-5),
#               metrics=['accuracy'])
#
# # Define Tensorboard as a Keras callback
# tensorboard = TensorBoard(
#     log_dir='.\logs',
#     histogram_freq=1,
#     write_images=True
# )
# keras_callbacks = [
#     tensorboard
# ]
#
# # Fit data to model
# model.fit(input_train, target_train,
#           batch_size=batch_size,
#           epochs=no_epochs,
#           verbose=verbosity,
#           # validation_split=validation_split,
#           validation_data = (target_train,target_test),
#           callbacks=keras_callbacks)
# # Generate generalization metrics
# score = model.evaluate(input_test, target_test, verbose=0)
# print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = normalize(x_train,axis = 1),normalize(x_test,axis = 1)

# (input_train,target_train),(input_test,target_test) = cifar10.load_data()

model = Sequential()
model.add(LSTM(128,input_shape = (x_train.shape[1:]),activation = 'relu',return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(128,activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(32,activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(10,activation = 'softmax'))
tensorboard = TensorBoard(
    log_dir='.\logs',
    histogram_freq=1,
    write_images=True
)
keras_callbacks = [
    tensorboard
]
opt = tf.keras.optimizers.Adam(lr = 1e-3,decay = 1e-5)
model.compile(optimizer = opt,loss = "sparse_categorical_crossentropy" , metrics=['accuracy'])
model.fit(x_train,y_train,epochs = 3,validation_data = (x_test,y_test),callbacks=keras_callbacks)
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')