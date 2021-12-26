import numpy as np #3-х слойная сеть
np.random.seed(1)
alpha = 0.2
hiden_size = 4

def relu(x):
    return (x>0)*x

def reluderiv(output):
    return output>0

streetlight=np.array([[1,0,1],
                      [0,1,1],
                      [0,0,1],
                      [1,1,1]])

walk_vs_stop=np.array([[1,1,0,0]]).T

weights_0_1=2*np.random.random((3,hiden_size))-1
weights_1_2=2*np.random.random((hiden_size,1))-1

for iteration in range (60):
    layer_2_error=0
    for i in range(len(streetlight)):
        layer_0 = streetlight[i:i+1] #берём каждую строку по отдельности
        layer_1 = relu(np.dot(layer_0,weights_0_1))
        layer_2 = np.dot(layer_1,weights_1_2)

        layer_2_error +=np.sum((layer_2-walk_vs_stop[i:i+1])**2) #среднеквадратичная ошибка

        layer_2_delta = (layer_2 - walk_vs_stop[i:i+1])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)*reluderiv(layer_1)

        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)

    if (iteration % 10 ==9):
        print("Error: ", layer_2_error)
