import numpy as np
import matplotlib.pylab as plt
from numpy.linalg import inv
import time

start = time.clock()

x = 0.1
y = 0.75
learning_rate = 1

w = np.array([x,y])

w_x = []
w_y = []
energy_matrix = []

while ((w[0]+w[1])<1) and (w[0]>0) and (w[1]>0):
    energy = - np.log(1-w[0]-w[1]) - np.log(w[0]) - np.log(w[1])
    energy_matrix.append(energy) 
    
    w_x.append(w[0])
    w_y.append(w[1])
    
    grad_x = 1/(1-w[0]-w[1]) - 1/(w[0])
    grad_y = 1/(1-w[0]-w[1]) - 1/(w[1]) 
    gradient = np.array([grad_x, grad_y])
    
    hessian_x1 = 1/((1-w[0]-w[1])*(1-w[0]-w[1])) + 1/(w[0]*w[0])
    hessian_y2 = 1/((1-w[0]-w[1])*(1-w[0]-w[1])) + 1/(w[1]*w[1])
    hessian_xy = 1/(1- w[0]-w[1]) * (1- w[0]-w[1])
    hessian = np.array([[hessian_x1, hessian_xy],[hessian_xy, hessian_y2]])
    
    update = learning_rate * np.matmul(inv(hessian), gradient)
    if np.linalg.norm(w - np.subtract(w,update)) < 0.001:
        break
    else:
        w = np.subtract(w,update)

end = time.clock()
print ("Time taken for Newton's method: ", round((end-start), 4))
plt.title("Gradient Descent Function")
plt.plot(w_x, w_y,'bo-')
plt.show()
plt.title("Energy Function Graph")
plt.plot(energy_matrix,'go-')
plt.show()

