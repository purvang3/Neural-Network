import numpy as np
import matplotlib.pyplot as plt

w0 = np.random.uniform(low=-0.25,high=0.25)
w1 = np.random.uniform(low=-1,high=1)
w2 = np.random.uniform(low=-1,high=1)
vectors = np.zeros((100,2))

for i in range(0,100):
	vectors[i] = np.random.uniform(low=-1,high=1,size=(2,))
	
add = np.ones((100,1))
newvectors = np.concatenate([add,vectors],axis=1)
ws = np.array((w0,w1,w2))
S = np.matmul(newvectors,ws)
S1 = np.where(S>=0)[0]
S2 = np.where(S<0)[0]

GR = vectors[S1]
LS = vectors[S2]
X = []
Y = []

for i in GR:
	X.append(i[0])
	Y.append(i[1])
nx = []
ny = []

for i in LS:
	nx.append(i[0])
	ny.append(i[1])
dx = np.linspace(-1,1)

if ws[2]>=0:
	dy = (ws[0]+ws[1]*dx)/(-1*ws[2])
else:
	dy = (ws[0]+ws[1]*dx)/abs(ws[2])
	
plt.plot(dx,dy,'r-',label='Decision boundary')
plt.plot(X,Y,'y*',label='S1')
plt.plot(nx,ny,'bo',label='S2')
plt.title('Classified Points');
plt.legend(['Decision boundary','S1','S2'])
plt.show()

classes = []
for i in range(len(newvectors)):
	if i in S1:
		classes.append(1)
	else:
		classes.append(0)
		
def step(x):
	if x>=0:
		return 1
	else:
		return 0

def missclassified(ws,newvectors,classes):
		dot = np.matmul(newvectors,ws)
		newclasses = []
		for i in dot:
			newclasses.append(step(i))
		er = np.array(classes) - np.array(newclasses)
		return sum(er!=0)
		
iw = np.random.uniform(low=-1,high=1,size=(3,))


def PTA(learning_rate,iw,classes,newvectors):
	fw = iw.copy()
	misclassnumb = [missclassified(iw,newvectors,classes)]
	temp = missclassified(iw,newvectors,classes)
	while temp>0:
		for i in range(len(newvectors)):
			prod = np.dot(newvectors[i],fw)
			out = step(prod)
			fw = fw + learning_rate*newvectors[i]*(classes[i]-out)
		temp = missclassified(fw,newvectors,classes)
		misclassnumb.append(temp)
		plt.plot(list(range(len(misclassnumb))),misclassnumb)
		plt.xlabel('Epoch Number')
		plt.ylabel('Number of misclassifications')
		plt.title('')
		plt.show()
		print("Number of epochs needed to converge: " + str(len(misclassnumb)))
		print("New ws:")
		print(fw)
	return fw

fw = PTA(1,iw,classes,newvectors)
fw = PTA(10,iw,classes,newvectors)
fw = PTA(0.1,iw,classes,newvectors)


iivec = np.random.uniform(-1,1,(1000,2))
add = np.ones((1000,1))
invec = np.concatenate([add,iivec],axis=1)
nclasses = np.matmul(invec,ws)
nS1 = np.where(nclasses>=0)[0]
nS2 = np.where(nclasses<0)[0]
GR = iivec[nS1]
LS = iivec[nS2]
X = []
Y = []
for i in GR:
	X.append(i[0])
	Y.append(i[1])
	
nx = []
ny = []
for i in LS:
	nx.append(i[0])
	ny.append(i[1])
dx = np.linspace(-1,1)

if ws[2]>=0:
	dy = (ws[0]+ws[1]*dx)/(-1*ws[2])
else:
	dy = (ws[0]+ws[1]*dx)/abs(ws[2])
	
plt.plot(dx,dy,'r-',label='Decision boundary')
plt.plot(X,Y,'y*',label='S1')
plt.plot(nx,ny,'b*',label='S2')
plt.legend(['Decision boundary','S1','S2'])
plt.show()

classes = []
for i in range(len(invec)):
	if i in nS1:
		classes.append(1)
else:
		classes.append(0)
		
fw = PTA(1,iw,classes,invec)
fw = PTA(10,iw,classes,invec)
fw = PTA(0.1,iw,classes,invec)
