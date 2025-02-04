
import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))
def sigmoid_d(x):
	o = sigmoid(x)
	return o*(1-o)
SIZE = 100

class NN:
	def __init__ (self,learning_rate=0.001):
		# Layer 1
		self.w_1 = np.random.rand(2,10)
		self.b_1 = np.zeros((1,10))
		self.z1 = np.zeros((1,10))
		# a1 neuron
		self.a1 = np.zeros((1,10))

		self.dw_1 = np.zeros((2,10))

		# Layer 2
		self.w_2 = np.random.rand(10,1)
		self.b_2 = np.zeros((1,1))
		self.z2 = np.zeros((1,1))
		# a2 neuron
		self.a2 = np.zeros((1,1))

		self.dw_2 = np.zeros((1,10))
		self.learning_rate = learning_rate

	def forward(self,x):
		self.z1 = x.dot(self.w_1)
		self.a1= sigmoid(self.z1+self.b_1)
		self.z2 = self.a1 @ self.w_2
		self.a2 = sigmoid(self.z2+self.b_2)
		return self.a2

	def cost(self,x,y):
		# error in output
		self.forward(x)
		return (1/2)*(self.a2-y)**2



	def backwards(self,x,target):
		"""
		@params:
		x - input
		target - true y value
		return gradiant
		"""

		# cost = 1/2*(target-output)^2
		# dC/da2
		# dcost/dtarget = output-target
		# Always vectors of the same size
		dC = self.a2-target

		# Sigmoid derivitive of the output from hidden layer 
		# da2/dz2 =
		dz2 = sigmoid_d(self.z2+self.b_2)

		# Chain rule, dC/dz2 = dC/da2*da2/dz2
		# dC/da2 known da2/dz2= sigmoid_d(self.z1+self.b_2) 
		# dC/dz2 = delta2 
		# error in hidden layer
		delta2 = dC*da2 

		# dC/db2 has the same derivitive as dC/dz2 = delta*1
		# dC/db2 = dC/dz2*dz2/db2
		# z2 = a1 @ w_2 + b_2
		# dz2 = d(a1 @ w_2 + b_2)/db2 = 1
		db2 = np.sum(delta2,axis=0,keepdims=True)

		# because z2 = a1*w_2+b_2
		da1 = sigmoid_d(self.z1+self.b_1)
		
		# Error in weights
		# dC/dw2
		dw2 = self.a1.T @ delta2

		# Chain rule continued dC/dz1=dC/da2*da2/dz2*dz2/da1*da1/dz1
		# da1 = da1/dz1 sigmoid derivitive

		# z = x*w1+b
		# dz1/dw1 = x
		# error in layer 1
		# dC/dz1 = delta 1
		# dC/dz1 = dC/dz2 * dC
		# propagate down through w_2
		# z1 = x*w_1+b1

		# C = a2(z2) = a2(a1 @ w_2 + b1)
		# dC/dz1 = dC/dz2*dz2/da1*da1/dz1 = 
		# dC/dz2 = delta2
		# dz2/da1 = w_2
		# a1=sigmoid(z_1 + b_1)
		# da1/dz1 = sigmoid'(z_1 + b_1)*d(z_1)/dz_1=sigmoid'(z_1+b_1)*1

		delta1 = delta2 @ self.w_2.T * da1
		# Notice the same derivitive is true for b1
		# dC/db1=dC/da2*da2/dz2*dz2/da1*da1/db1

		db1 = np.sum(delta1,axis=0,keepdims=True)
		
		# dC/dw1 = dC/z1*dz1/d_w1
		# z1 = x @ w_1
		# a1 = sigmoid(z1) 
		# w_1 is a function of z_1
		# 
		dw1 = x.T @ delta1 
		
		
		return dw1,db1,dw2,db2


	def train(self,x,target):
		self.forward(x)
		dw1,db1,dw2,db2 = self.backwards(x,target)
		self.w_1 -= dw1*self.learning_rate
		self.w_2 -= dw2*self.learning_rate
		self.b_1 -= db1*self.learning_rate
		self.b_2 -= db2*self.learning_rate
		return self.cost(x,target)

EPOCHS = 1000
if __name__ == "__main__":
	nn = NN(learning_rate=0.5)	
	
	X = np.array([np.random.random_integers(low=0,high=1,size=SIZE),np.random.random_integers(low=0,high=1,size=SIZE)])
	Y = np.array([[X[:, i][0] ^ X[:, i][1]] for i in range(X.shape[1])])
	for epoch in range(EPOCHS):
	    for i in range(X.shape[1]):
	        x = np.reshape(X[:, i], (1, 2))
	        cost_value = nn.train(x, Y[i])
	    # Maybe print the cost after the epoch, for the entire dataset
	    if epoch % 100 == 0:
	        # Evaluate average cost on all data
	        total_cost = 0
	        for i in range(X.shape[1]):
	            x = np.reshape(X[:, i], (1, 2))
	            total_cost += nn.cost(x, Y[i])[0, 0]
	        avg_cost = total_cost / X.shape[1]
	        print(f"Epoch {epoch}, average cost = {avg_cost}")

	while(True):
		a = input()
		b = input()
		print(nn.forward(np.array([int(a),int(b)])))