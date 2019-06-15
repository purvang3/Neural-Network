import struct
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

#Weight Function for generating weights
def weightfunc():
    i_weight = np.zeros((10,784))
    for i in range(0,10):
        x = (1-(-1))*random.sample(784) - 1
        i_weight[i] = x
    weight_matrix=np.matrix(np.array(i_weight))
    return weight_matrix

##Function to train images for Multicategory TrainPTA
##def TrainPTA(N1,eta,epsilon,weight_matrix):
##    # Data read from the Label files
##    label_data= open('train-labels-idx1-ubyte','rb')
##    struct.unpack('>I',label_data.read(4))[0]
##    struct.unpack('>I',label_data.read(4))[0]
##    desired=[]
##    for j in range(N1): 
##        desired.append(struct.unpack('>B',label_data.read(1))[0])
##    flag=1
##    error=0
##    epoch=0
##    epoch_arr=[]
##    error_arr=[]
##    while(flag==1):
##        error=0
##        img_data= open('train-images-idx3-ubyte','rb')
##        img_data.read(4)
##        img_data.read(4)
##        img_data.read(4)
##        img_data.read(4)
##        for k in range(N1):
##            x1=[]
##            for l in range(784):
##                x1.append(struct.unpack('>B',img_data.read(1))[0])
##            matrix_inputs=np.matrix(np.array(x1))
##            matrix_inputs_transpose=np.transpose(matrix_inputs)
##            dot=np.dot(weight_matrix,matrix_inputs_transpose)
##            actval=np.argmax(dot)
##            new_mat=np.zeros((10,1))
##            if(desired[k]!=actval):
##                new_mat[actval]=-1
##                new_mat[desired[k]]=1
##                error=error+1
##                weight_matrix=weight_matrix + np.dot((eta*new_mat),matrix_inputs)
##        epoch+=1
##        error_arr.append(error)
##        epoch_arr.append(epoch)
##        if(epoch==70): break
##        print("Epoch =  ",epoch," Error = ",error)
##        if((error/N1)>epsilon):
##            flag=1
##        else:
##            flag=0
##    error_percentage = (error/N1)*100
##    print("Error Percentage = ",error_percentage,"%")
##    print("Epoch=",epoch)
##    print("Error=",error)
##    plt.title("Graph for Epoch Vs Miscount")
##    plt.xlim(0,epoch+1)
##    plt.plot(epoch_arr,error_arr,'o-')
##    plt.xlabel('Epoch Number')
##    plt.ylabel('Number of misclassifications')
##    plt.show()
##    return(weight_matrix)
##
##Function to test images for Multicategory TrainPTA
##def TestPTA(N2,weight_matrix):
##    error_percentage=0
##    label_data= open('t10k-labels-idx1-ubyte','rb')
##    struct.unpack('>I',label_data.read(4))[0]#Reading first two bytes
##    struct.unpack('>I',label_data.read(4))[0]
##    desired=[]
##    for j in range(N2): 
##        desired.append(struct.unpack('>B',label_data.read(1))[0])
##    error=0
##    img_data= open('t10k-images-idx3-ubyte','rb')
##    img_data.read(4)
##    img_data.read(4)
##    img_data.read(4)
##    img_data.read(4)
##    for k in range(N2):
##        x1=[]
##        for l in range(0,784):
##            x1.append(struct.unpack('>B',img_data.read(1))[0])
##        matrix_inputs=np.matrix(np.array(x1))
##        matrix_inputs_transpose=np.transpose(matrix_inputs)
##        dot=np.dot(weight_matrix,matrix_inputs_transpose)
##        actval=np.argmax(dot)
##        new_mat=np.zeros((10,1))
##        if(desired[k]!=actval):
##            new_mat[actval]=-1
##            new_mat[desired[k]]=1
##            error=error+1
##    error_percentage = (error/N2)*100
##    print("Error Percentage = ",error_percentage,"%")
##
##print("Executing all possible combinations")
##print("Program Starts")
##weight_glob=weightfunc()
##
##print("Part f(1)")
##weight_2=TrainPTA(50,1,0,weight_glob)
##TestPTA(50,weight_2)
##
##print("Part f(2)")
##weight_3=TrainPTA(50,1,0,weight_glob)
##TestPTA(10000,weight_3)
##
##print("Part g(1)")
##weight_4=TrainPTA(1000,1,0,weight_glob)
##TestPTA(1000,weight_4)
##
##print("Part g(2)")
##weight_5=TrainPTA(1000,1,0,weight_glob)
##TestPTA(10000,weight_5)
##
##print("Part h")
##weight_1=TrainPTA(60000,1,0,weight_glob)
##TestPTA(10000,weight_1)
##
##print("Part i(1) Set")
##weight_glob1=weightfunc()
##weight_6=TrainPTA(60000,1,0.10,weight_glob1)
##TestPTA(10000,weight_6)
##
##print("Part i(2) Set")
##weight_glob2=weightfunc()
##weight_7=TrainPTA(60000,1,0.11,weight_glob2)
##TestPTA(10000,weight_7)
##
##print("Part i(3) Set")
##weight_glob3=weightfunc()
##weight_8=TrainPTA(60000,1,0.09,weight_glob3)
##TestPTA(10000,weight_8)
