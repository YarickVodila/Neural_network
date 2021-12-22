import numpy as np
import random
def sigmoid(x):
    # Функция активации sigmoid:: f(x) = 1 / (1 + e^(-x))
    #print(1 / (1 + np.exp(-x)))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Производная от sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

class Network:
    E = 0.1 #скорость обучения
    A =0.3  #момент

    delta_w1 = 0
    delta_w2 = 0
    delta_w3 = 0
    delta_w4 = 0
    delta_w5 = 0
    delta_w6 = 0
    delta_b1 = 0
    delta_b2 = 0
    delta_b3 = 0
    def __init__(self):
        # Вес
        self.w1 = random.uniform(-1,1)
        self.w2 = random.uniform(-1,1)
        self.w3 = random.uniform(-1,1)
        self.w4 = random.uniform(-1,1)
        self.w5 = random.uniform(-1,1)
        self.w6 = random.uniform(-1,1)

        # Смещения

        self.b1 = 1
        self.b2 = 1


    def result(self,x,y):
        h1 = sigmoid(self.w1 * x + self.w3 * y + self.b1)
        h2 = sigmoid(self.w2 * x + self.w4 * y + self.b2)
        out = sigmoid(self.w5 * h1 + self.w6 * h2 )
        return out


    def tranning(self):
        global res, res2
        for epoch in range(1000):
            for i in range(len(res)):
                h1 = sigmoid(self.w1 * res[i][0] + self.w3 * res[i][1] + self.b1)
                h2 = sigmoid(self.w2 * res[i][0] + self.w4 * res[i][1] + self.b2)
                out = sigmoid(self.w5 * h1 + self.w6 * h2 )
                error = (res2[i] - out)**2 # Ошибка
                #print(error*100)

                delta_out = (res2[i]-out)*deriv_sigmoid(out)

                delta_h1 = deriv_sigmoid(h1) * (self.w5 * delta_out)
                GRAD_w5 = h1*delta_out
                self.delta_w5 = self.E * GRAD_w5 + self.delta_w5 * self.A
                self.w5 = self.w5 - GRAD_w5 * self.E*h1

                delta_h2 = deriv_sigmoid(h2) * (self.w6 * delta_out)
                GRAD_w6 = h2 * delta_out
                self.delta_w6 = self.E * GRAD_w6 + self.delta_w6 * self.A
                self.w6 = self.w6 - GRAD_w5 * self.E*h2

                GRAD_w1 = res[i][0] * delta_h1
                GRAD_w2 = res[i][0] * delta_h2
                GRAD_w3 = res[i][1] * delta_h1
                GRAD_w4 = res[i][1] * delta_h2

                self.delta_w1 = self.E * GRAD_w1 + self.delta_w1 * self.A
                self.delta_w2 = self.E * GRAD_w2 + self.delta_w2 * self.A
                self.delta_w3 = self.E * GRAD_w3 + self.delta_w3 * self.A
                self.delta_w4 = self.E * GRAD_w4 + self.delta_w4 * self.A

                self.w1 = self.w1 - GRAD_w1 * self.E
                self.w2 = self.w2 - GRAD_w2 * self.E
                self.w3 = self.w3 - GRAD_w3 * self.E
                self.w4 = self.w4 - GRAD_w4 * self.E




'''
res=[
    [-2, -1],    # Alice
    [25, 6],     # Bob
    [17, 4],     # Charlie
    [-15, -6], # Diana
]




res2=[1,0,0,1] #Даннаые которые должны получить

'''
res=[[0,0],
     [1,1],
    [2,2],
    [3,3],
    [4,4],
    [5,5],
    [6,6],

     [1,2],
     [1,0]]

res2=[1,1,1,1,1,1,1,0,0] #Даннаые которые должны получить



network = Network()
network.tranning()
print("Результаты \n")
'''
for i in range(len(res)):
    print(network.result(res[i][0],res[i][1]))
    print("\n")
'''


print(network.result(3,0))
print(network.result(2,2))