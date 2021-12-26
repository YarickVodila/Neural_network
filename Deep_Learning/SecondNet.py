import numpy as np

w=np.array([0.5,0.48,-0.7])
alpha=0.1
streetlight=np.array([[1,0,1],
                      [0,1,1],
                      [0,0,1],
                      [0,1,1],
                      [0,1,1],
                      [1,0,1]])
walk_vs_stop=np.array([0,1,0,1,1,0])
input=streetlight[0]
goal_pred=walk_vs_stop[0]
print('Веса до ',w)
for i in range (40):
    error_ALL=0
    for row in range(len(walk_vs_stop)):
        input=streetlight[row]
        goal_pred=walk_vs_stop[row]
        pred=input.dot(w)
        error=(goal_pred-pred)**2
        error_ALL+=error
        delta=pred-goal_pred
        w-=(alpha*input*delta)
        '''
        print('pred: ',pred)
    print('Error: ',error_ALL)'''
print('Веса после ',w)
