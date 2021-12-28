import sys,numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

images, labels = (x_train[0:1000].reshape(1000,28*28),255,y_train[0:1000])

one_hot_labels = np.zeros((len(labels),10))

for i,l in enumerate (labels):
    one_hot_labels[i][l] =1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test),28*28)/255
test_labels = np.zeros((len(y_test),10))
for i,l in enumerate (y_test):
    test_labels[i][l]=1

np.random.seed(1)
relu = lambda x:(x>=0)*x #функция активации relu (Возвращает x если x>0,иначе 0)
relu2deriv = lambda x:x>=0 #производная от relu (Возвращает 1 если x>0,иначе 0)

alpha, iterations, hidden_size, pixels_per_image, num_labels = \
    (0.005,350,40, 784, 10)
weight_0_1 = 0.2*np.random.random((pixels_per_image,hidden_size))-0.1
print(weight_0_1)
weight_1_2 = 0.2*np.random.random((hidden_size,num_labels))-0.1
print(weight_1_2)