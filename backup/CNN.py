
import numpy as np
import os
from keras.datasets import mnist
import matplotlib.pyplot as plt
import sys
from tempfile import TemporaryFile
outfile = TemporaryFile()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


FILE_NAME = 'nn-cnn.npz'
INPUT_SIZE = 676 
#INPUT_SIZE = 784
HIDDEN_LAYER_SIZE = 80
WEIGHT_DISTRIBUTION = 16 #(80/5)
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

def show_image(image):
	plt.imshow(image)
	plt.show()


class NN:
	# Convolutional filter size 3x3
	
	kernel = np.ones((3,3))
	kernel_left = np.array([[1,0,-1],
							[1,0,-1],
							[1,0,-1]])
	kernel_right = np.array([[-1,0,1],
							[-1,0,1],
							[-1,0,1]])
	kernel_up = np.array([[1,1,1],
							[0,0,0],
							[-1,-1,-1]])
	kernel_down = np.array([[-1,-1,-1],
							[0,0,0],
							[1,1,1]])

	def unflatten_convolution(self,X):
		return X.reshape((self.cluster_shape[0],self.cluster_shape[1]))
	def flatten_convolution(self,X):
		return X.reshape(X.shape[-1]**2)

	def __init__ (self,learning_rate=0.1):
		# Layer 1
		self.w_e = np.random.rand(INPUT_SIZE,WEIGHT_DISTRIBUTION) * np.sqrt(2 / 784)
		self.w_l = np.random.rand(INPUT_SIZE,WEIGHT_DISTRIBUTION) * np.sqrt(2 / 784)
		self.w_r = np.random.rand(INPUT_SIZE,WEIGHT_DISTRIBUTION) * np.sqrt(2 / 784)
		self.w_u = np.random.rand(INPUT_SIZE,WEIGHT_DISTRIBUTION) * np.sqrt(2 / 784)
		self.w_d = np.random.rand(INPUT_SIZE,WEIGHT_DISTRIBUTION) * np.sqrt(2 / 784)


		#self.w_1 = np.random.rand(INPUT_SIZE,HIDDEN_LAYER_SIZE) * np.sqrt(2 / 784)
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

		self.cluster_shape = None

		self.ce = np.zeros((1,WEIGHT_DISTRIBUTION))
		self.cl = np.zeros((1,WEIGHT_DISTRIBUTION))
		self.cr = np.zeros((1,WEIGHT_DISTRIBUTION))
		self.cu = np.zeros((1,WEIGHT_DISTRIBUTION))
		self.cd = np.zeros((1,WEIGHT_DISTRIBUTION))

	def forward(self,x):
		ce = self.conv_2d(x,self.kernel)
		cl = self.conv_2d(x,self.kernel_left)
		cr = self.conv_2d(x,self.kernel_right)
		cu = self.conv_2d(x,self.kernel_up)
		cd = self.conv_2d(x,self.kernel_down)
		self.cluster_shape = ce.shape
		self.ce = self.flatten_convolution(ce).reshape(1,INPUT_SIZE)
		self.cl = self.flatten_convolution(cl).reshape(1,INPUT_SIZE)
		self.cr = self.flatten_convolution(cr).reshape(1,INPUT_SIZE)
		self.cu = self.flatten_convolution(cu).reshape(1,INPUT_SIZE)
		self.cd = self.flatten_convolution(cd).reshape(1,INPUT_SIZE)

		self.ze = self.ce.dot(self.w_e)
		self.zl = self.cl.dot(self.w_l)
		self.zr = self.cr.dot(self.w_r)
		self.zu = self.cu.dot(self.w_u)
		self.zd = self.cd.dot(self.w_d)
		# Stack layers
		self.z1 = np.array([self.ze,self.zl,self.zr,self.zu,self.zd]).reshape((1,HIDDEN_LAYER_SIZE))

		self.a1= ReLU(self.z1+self.b_1)
		self.z2 = self.a1 @ self.w_2
		#self.a2 = sigmoid(self.z2+self.b_2)
		self.a2 = self.z2
		return self.a2

	def conv_2d(self,A,kernel,stride=1):
		x = int(np.floor((A.shape[0]-(kernel.shape[0]))/stride)+1)
		y = int(np.floor((A.shape[1]-(kernel.shape[1]))/stride)+1)
		out = np.ones((x,y))
		c = lambda x,y: np.einsum('ij,ij->',A[x:x+kernel.shape[0],y:y+kernel.shape[1]],kernel)
		for i in range(out.shape[0]): 
			for j in range(out.shape[1]):
				out[i,j] = c(i*stride,j*stride)
		return out


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

		# cost = -log(softmax(target)*y))
		# dcost/dtarget = softmax(output)-target
		dC = self.loss_deritivitve(target)
		#dC = self.a2-target

		# Sigmoid derivitive of the output from hidden layer 
		# da2/dz2 =

		# Chain rule, dC/dz2 = dC/da2*da2/dz2
		#da2 = 1.
		delta2 = dC

		# dC/db2 has the same derivitive as dC/dz2 = delta*1
		# dC/db2 = dC/dz2*dz2/db2
		db2 = np.sum(delta2,axis=0,keepdims=True)

		da1 = dRelu(self.z1+self.b_1)
		
		# Error in weights
		# dC/dw2
		dw2 = self.a1.T @ delta2
		
		# C = a2(z2) = a2(a1 @ w_2 + b1)
		# dC/dz1 = dC/dz2*dz2/da1*da1/dz1 = 
		delta1 = delta2 @ self.w_2.T * da1

		# dC/db1=dC/da2*da2/dz2*dz2/da1*da1/db1

		db1 = np.sum(delta1,axis=0,keepdims=True)
		
		# delta1 = error in layer 1 
		# layer 1 = z1 = [ze,zl,zr,zu,zd]
		delta1_ce = delta1[:,0:WEIGHT_DISTRIBUTION]
		delta1_cl = delta1[:,WEIGHT_DISTRIBUTION:WEIGHT_DISTRIBUTION*2]
		delta1_cr = delta1[:,WEIGHT_DISTRIBUTION*2:WEIGHT_DISTRIBUTION*3]
		delta1_cu = delta1[:,WEIGHT_DISTRIBUTION*3:WEIGHT_DISTRIBUTION*4]
		delta1_cd = delta1[:,WEIGHT_DISTRIBUTION*4:WEIGHT_DISTRIBUTION*5]

		# dC/dw1 = dC/z1*dz1/d_w1
		dwe = self.ce.T @ delta1_ce
		dwl = self.cl.T @ delta1_cl
		dwr = self.cr.T @ delta1_cr
		dwu = self.cu.T @ delta1_cu
		dwd = self.cd.T @ delta1_cd

		# full gradiant
		return dwe,dwl,dwr,dwu,dwd,db1,dw2,db2


	def train(self,x,target):
		self.forward(x)
		dwe,dwl,dwr,dwu,dwd,db1,dw2,db2 = self.backwards(x,target)
		self.w_e -= dwe*self.learning_rate
		self.w_l -= dwl*self.learning_rate
		self.w_r -= dwr*self.learning_rate
		self.w_u -= dwu*self.learning_rate
		self.w_d -= dwd*self.learning_rate
		self.w_2 -= dw2*self.learning_rate
		self.b_1 -= db1*self.learning_rate
		self.b_2 -= db2*self.learning_rate
		return self.cost(target)

	def array_prediction(self,x):
		self.forward(x)
		values = self.a2
		expected_value = np.argmax(values)
		x = np.sum(values) 
		y = 1/x
		values = softmax(values[0]*y)
		values = 100*values
		print(values)
		print(f"Predicted value: {expected_value}, with a {values[expected_value]}% prediction certainty\n")

	def predict_value(self,x):
		self.forward(x)
		return np.argmax(self.a2)



	def save_network(self):
		print("Saving network")
		np.savez(FILE_NAME,self.w_e,self.w_l,self.w_r,self.w_u,self.w_d,self.w_2,self.b_1,self.b_2)
		_ = outfile.seek(0)
	def load_network(self):
		npzfile = np.load(FILE_NAME)
		self.w_e = npzfile['arr_0']
		self.w_l = npzfile['arr_1']
		self.w_r = npzfile['arr_2']
		self.w_u = npzfile['arr_3']
		self.w_d = npzfile['arr_4']
		self.w_2 = npzfile['arr_5']
		self.b_1 = npzfile['arr_6']
		self.b_2 = npzfile['arr_7']


	def train_network(self,train_X,train_Y,epochs):
		for epoch in range(epochs):
			indices = np.random.permutation(len(train_X))
			train_X = train_X[indices]
			train_Y = train_Y[indices]
			for start_idx in range(0, len(train_X), 1):

				x_batch = train_X[start_idx]
				y_batch = train_Y[start_idx]
	
	
				# reshape x_batch => (batch_size, 784)
				# maybe one-hot y_batch => (batch_size, 10)
	
				cost_val = nn.train(x_batch, y_batch)
			print(f"Epoch {epoch} finished, last mini-batch cost {cost_val.mean()}")

	def test_nn_accuracy(self,test_X,test_Y):
		sample_size = len(test_X)
		correct_predictions = 0
		for i in range(sample_size):
			self.predict_value(test_X[i])
			if np.argmax(test_Y[i]) == self.predict_value(test_X[i]):
				correct_predictions += 1
		return correct_predictions/sample_size

def flatten_image(X):
	return X.reshape(X.shape[0], X.shape[-1]**2)

def unflatten_image(X,dim):
	return X.reshape((dim,28,28))

SIZE = 2
TEST_SAMPLES = 10

def test_nn_on_samples(nn,test_X,test_Y,size=SIZE):
	for i in range(TEST_SAMPLES):
			step = np.random.randint(0,10)*SIZE
			x = test_X[i*TEST_SAMPLES+step]
			#image = unflatten_image(test_X[i*TEST_SAMPLES+step],1)[0]

			number = np.argmax(test_Y[i*TEST_SAMPLES+step])
			print(f"the number is {number} - index {i*TEST_SAMPLES+step}")
			nn.array_prediction(x)
			#nn.forward_print(x)
			plt.imshow(x)
			plt.show()


EPOCHS = 200
batch_size = 10

TRAIN_START = 3000
TRAIN_SIZE = 1000

TRAIN = False


if __name__ == "__main__":
	(train_X,train_Y), (test_X,test_Y) = mnist.load_data()

	nn = NN(learning_rate=0.0001)
	#train_X = flatten_image(train_X)
	train_X = train_X / 255.0
	Y = train_Y
	train_Y = np.array([[1 if train_Y[x] == i else 0 for i in range (10)] for x in range(train_Y.shape[0])])
	test_Y = np.array([[1 if test_Y[x] == i else 0 for i in range (10)] for x in range(test_Y.shape[0])])
	#for epoch in tqdm(range(EPOCHS)):
	
	if TRAIN:
		nn.load_network()
		nn.train_network(train_X[TRAIN_START:TRAIN_START+TRAIN_SIZE],train_Y[TRAIN_START:TRAIN_START+TRAIN_SIZE],EPOCHS)
		nn.save_network()

	else:
		nn.load_network()
	#else:
	#	#nn.forward(train_X[0])
	#	#nn.backwards()
	#	nn.train(train_X[0],train_Y[0])
	#	nn.train(train_X[1],train_Y[1])
	#	#nn.conv_2d(train_X)

	test_nn_on_samples(nn,test_X,test_Y)
	#print(nn.test_nn_accuracy(test_X[2000:3000],test_Y[2000:3000]))
