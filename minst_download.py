import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
(train_X,train_Y), (test_X,test_Y) = mnist.load_data()
index = 10


for i in range(30,39):
		image = train_X[i]
		number = train_Y[i]
		print(number)
		plt.imshow(image)
		plt.show()