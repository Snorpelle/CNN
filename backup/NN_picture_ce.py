
import numpy as np
import os
from keras.datasets import mnist
import matplotlib.pyplot as plt
from tqdm import tqdm
from tempfile import TemporaryFile
outfile = TemporaryFile()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


FILE_NAME = 'nn-ce.npz'
INPUT_SIZE = 784
HIDDEN_LAYER_SIZE = 64
OUTPUT_SIZE = 10


# Sigmoid function & derivitive
def sigmoid(x):
	return 1/(1+np.exp(-x))
def sigmoid_d(x):
	o = sigmoid(x)
	return o*(1-o)

# ReLU function & derivitive
def ReLU(x):
	return x * (x > 0)
def dRelu(z):
    return np.where(z <= 0, 0, 1)

# Softmax
def softmax(x):
	e = np.exp(x)
	sm = e / np.sum(e, axis=-1, keepdims=True)
	return sm

# Cross entropy 
def cross_entropy(x,y):
	H = -1/OUTPUT_SIZE*np.log(softmax(x))*y
	#H= -np.mean(np.sum(y * np.log(softmax(x)), axis=-1))
	return H


class NN:

	def __init__ (self,learning_rate=0.1):
		# Layer 1
		self.w_1 = np.random.rand(INPUT_SIZE,HIDDEN_LAYER_SIZE) * np.sqrt(2 / 784)
		self.b_1 = np.zeros((1,HIDDEN_LAYER_SIZE))
		self.z1 = np.zeros((1,HIDDEN_LAYER_SIZE))
		# a1 neuron
		self.a1 = np.zeros((1,HIDDEN_LAYER_SIZE))

		self.dw_1 = np.zeros((INPUT_SIZE,HIDDEN_LAYER_SIZE))

		# Layer 2
		self.w_2 = np.random.rand(HIDDEN_LAYER_SIZE,OUTPUT_SIZE) * np.sqrt(2 / 784)
		self.b_2 = np.zeros((1,OUTPUT_SIZE))
		self.z2 = np.zeros((1,OUTPUT_SIZE))
		# a2 neuron
		self.a2 = np.zeros((1,OUTPUT_SIZE))

		self.dw_2 = np.zeros((HIDDEN_LAYER_SIZE,OUTPUT_SIZE))
		self.learning_rate = learning_rate

	def forward(self,x):
		self.z1 = x.dot(self.w_1)
		self.a1= ReLU(self.z1+self.b_1)
		self.z2 = self.a1 @ self.w_2
		#self.a2 = sigmoid(self.z2+self.b_2)
		self.a2 = self.z2
		return self.a2

	def cost(self,target):
		# Cross entropy loss
		return cross_entropy(self.a2,target)

	def loss_deritivitve(self,y):
		# Cross entropy derivitve
		return softmax(self.a2)-y


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
		dC = self.loss_deritivitve(target)
		#dC = self.a2-target

		# Sigmoid derivitive of the output from hidden layer 
		# da2/dz2 =
		#da2 = sigmoid_d(self.z2+self.b_2)
		
		# Chain rule, dC/dz2 = dC/da2*da2/dz2
		# dC/da2 known da2/dz2= sigmoid_d(self.z1+self.b_2) 
		# dC/dz2 = delta2 
		# error in hidden layer
		#delta2 = dC*da2 
		delta2 = dC

		# dC/db2 has the same derivitive as dC/dz2 = delta*1
		# dC/db2 = dC/dz2*dz2/db2
		# z2 = a1 @ w_2 + b_2
		# dz2 = d(a1 @ w_2 + b_2)/db2 = 1
		db2 = np.sum(delta2,axis=0,keepdims=True)

		# because z2 = a1*w_2+b_2
		da1 = dRelu(self.z1+self.b_1)

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
		dw1 = x.T @ delta1 
		
		# full gradiant
		return dw1,db1,dw2,db2


	def train(self,x,target):
		self.forward(x)
		dw1,db1,dw2,db2 = self.backwards(x,target)
		self.w_1 -= dw1*self.learning_rate
		self.w_2 -= dw2*self.learning_rate
		self.b_1 -= db1*self.learning_rate
		self.b_2 -= db2*self.learning_rate
		return self.cost(target)

	def predict_value(self,x):
		self.forward(x)
		values = self.a2
		expected_value = np.argmax(values)
		x = np.sum(values) 
		y = 1/x
		values = softmax(values[0]*y)
		values = 100*values
		print(values)
		print(f"Predicted value: {expected_value}, with a {values[expected_value]}% prediction certainty\n")

	def save_network(self):
		print("Saving network")
		np.savez(FILE_NAME,self.w_1,self.w_2,self.b_1,self.b_2)
		_ = outfile.seek(0)
	def load_network(self):
		npzfile = np.load(FILE_NAME)
		self.w_1 = npzfile['arr_0']
		self.w_2 = npzfile['arr_1']
		self.b_1 = npzfile['arr_2']
		self.b_2 = npzfile['arr_3']


	def train_network(self,train_X,train_Y,epochs):
		for epoch in range(epochs):
			indices = np.random.permutation(len(train_X))
			train_X = train_X[indices]
			train_Y = train_Y[indices]
			for start_idx in range(0, len(train_X), batch_size):
				end_idx = start_idx + batch_size
				x_batch = train_X[start_idx:end_idx]
				y_batch = train_Y[start_idx:end_idx]
	
	
				# reshape x_batch => (batch_size, 784)
				# maybe one-hot y_batch => (batch_size, 10)
	
				cost_val = nn.train(x_batch, y_batch)
			print(f"Epoch {epoch} finished, last mini-batch cost {cost_val.mean()}")



def flatten_image(X):
	return X.reshape(X.shape[0], X.shape[-1]**2)

def unflatten_image(X,dim):
	return X.reshape((dim,28,28))

SIZE = 15
samples = 10

def test_nn_on_samples(nn,test_X,test_Y,size=SIZE):
	for i in range(samples):
			step = np.random.randint(0,10)*SIZE
			x = np.reshape(test_X[i*samples+step], (1,INPUT_SIZE))
			image = unflatten_image(test_X[i*samples+step],1)[0]
			number = np.argmax(test_Y[i*samples+step])
			print(f"the number is {number}")
			nn.predict_value(x)
			#nn.forward_print(x)
			plt.imshow(image)
			plt.show()


EPOCHS = 2000
batch_size = 10

if __name__ == "__main__":
	(train_X,train_Y), (test_X,test_Y) = mnist.load_data()

	nn = NN(learning_rate=0.0001)	
	train_X	 = flatten_image(train_X)
	train_X = train_X / 255.0
	Y = train_Y
	train_Y = np.array([[1 if train_Y[x] == i else 0 for i in range (10)] for x in range(train_Y.shape[0])])
	test_Y = np.array([[1 if test_Y[x] == i else 0 for i in range (10)] for x in range(test_Y.shape[0])])
	#for epoch in tqdm(range(EPOCHS)):
	TRAIN = False
	if TRAIN:
		nn.train_network(train_X,train_Y,EPOCHS)
		nn.save_network()
	else:
		nn.load_network()

	test_nn_on_samples(nn,test_X,test_Y)
	
