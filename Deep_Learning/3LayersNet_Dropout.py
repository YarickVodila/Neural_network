import numpy as np,sys #3-х слойная сеть c dropout регуляризацией
np.random.seed(1)
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

images= (x_train[0:1000].reshape(1000,28*28))
labels = (255,y_train[0:1000])
test_images = x_test.reshape(len(x_test),28*28)/255
test_labels = np.zeros((len(y_test),10))

def relu(x):
    return (x>0)*x

def reluderiv(output):
    return output>0
alpha, iterations, hidden_size = (0.005, 300, 100)
pixels_per_image, num_labels = (784, 10)

weights_0_1=2*np.random.random((pixels_per_image,hidden_size))-0.1
weights_1_2=2*np.random.random((hidden_size,num_labels))-0.1

for j in range (iterations):

    error, correct_cnt = (0.0,0)

    for i in range(len(images)):
        layer_0 = images[i:i+1]
        layer_1 = relu(np.dot(layer_0,weights_0_1))
        dropout_mask = np.random.randint(2,size=layer_1.shape)
        layer_1 *=dropout_mask*2
        layer_2 = np.dot(layer_1,weights_1_2)

        error += np.sum((labels[i:i+1] - layer_2)**2)
        correct_cnt +=int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        layer_2_delta = (labels[i:i+1] - layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)*reluderiv(layer_1)
        layer_1_delta *= dropout_mask

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if (j%10==0):
        test_error = 0.0
        test_correct_cnt = 0

        for i in range (len(test_images)):
            layer_0 = test_images[i:i+1]
            layer

