import tensorflow as tf
import os

from keras import layers
from tensorflow import keras
from tensorflow._api.v2.v2 import optimizers
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import numpy as np
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw)=datasets.mnist.load_data()
print(x_train_raw.shape)
num_class=10
y_train=keras.utils.to_categorical(y_train_raw,num_class)
y_test=keras.utils.to_categorical(y_test_raw,num_class)
print(y_train.shape)
plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train_raw[i])
plt.show()
x_train = x_train_raw.reshape(60000,784)
x_test=x_test_raw.reshape(10000,784)
x_train =x_train.astype('float32')/255
x_test =x_test.astype('float32')/255
model=keras.Sequential([
    layers.Dense('512',activation='relu',input_dim=784),
    layers.Dense('256',activation='relu'),
    layers.Dense('124',activation='relu'),
    layers.Dense(num_class,activation='softmax')
])
model.summary()
Optimazer = optimizers.Adam(0.001)
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=Optimazer,metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=128,epochs=10,verbose=1)
score=model.evaluate(x_test,y_test,verbose=1)
print('test loss',score[0])
print('test accuracy',score[1])
logdir='./models/'
if not os.path.exists(logdir):
    os.mkdir(logdir)
model.save(logdir+'final_dll_model.h5')
model.summary()