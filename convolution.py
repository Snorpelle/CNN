
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


def conv2d(A,kernel,stride=1):
    x = int(np.floor((A.shape[0]-(kernel.shape[0]))/stride)+1)
    y = int(np.floor((A.shape[1]-(kernel.shape[1]))/stride)+1)
    out = np.ones((x,y))
    c = lambda x,y: np.einsum('ij,ij->',A[x:x+kernel.shape[0],y:y+kernel.shape[1]],kernel)
    for i in range(out.shape[0]): 
         for j in range(out.shape[1]):
            out[i,j] = c(i*stride,j*stride)
    return out

def show_image(image):
    plt.imshow(image)
    plt.show()

(train_X,train_Y), (test_X,test_Y) = mnist.load_data()
#kernel = np.ones((4,4))
#kernel_left = np.array([[1,0,0,-1],
#                    [1,0,0,-1],
#                    [1,0,0,-1],
#                    [1,0,0,-1]])
#kernel_right = np.array([[-1,0,0,1],
#                        [-1,0,0,1],
#                        [-1,0,0,1],
#                        [-1,0,0,1]])
#kernel_up = np.array([[1,1,1,1],
#                    [0,0,0,0],
#                    [0,0,0,0],
#                    [-1,-1,-1,-1]])
#kernel_down = np.array([[-1,-1,-1,-1],
#                    [0,0,0,0],
#                    [0,0,0,0],
#                    [1,1,1,1]])

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


tX = train_X[87] / 255.0
print(tX)
show_image(tX)
#c_l = conv2d(tX,kernel_left,1)
#c_r = conv2d(tX,kernel_right,1)
#c_u = conv2d(tX,kernel_up,1)
#c_d = conv2d(tX,kernel_down,1)
#c_e = conv2d(tX,kernel,1)

c_l = conv2d(conv2d(tX,kernel_left,1),kernel_left,1)
c_r = conv2d(conv2d(tX,kernel_right,1),kernel_right,1)
c_u = conv2d(conv2d(tX,kernel_up,1),kernel_up,1)
c_d = conv2d(conv2d(tX,kernel_down,1),kernel_down,1)
c_e = conv2d(conv2d(tX,kernel,1),kernel,1)

show_image(c_l)
show_image(c_r)
show_image(c_u)
show_image(c_d)
show_image(c_e)
print("c_l")
print(c_l)
print("c_r")
print(c_r)
print("c_u")
print(c_u)
print("c_d")
print(c_d)
#print(c1.shape)




#A = np.array([[0, 1, 2, 4, 100], 
#            [5, 6, 7 , 8, 200],
#            [9, 10, 11 ,12, 300],
#            [-1, -2, -3 ,-4, -5],
#            [-1, -2, -3 ,-4, -5]])
#
#c1 = conv2d(A,kernel,1)












def convolve2d_valid(A, kernel):
    # Dimensions
    H, W = A.shape
    kH, kW = kernel.shape
    outH = H - kH + 1
    outW = W - kW + 1
    
    # Build a view of A as all submatrices of shape (kH, kW)
    # Using np.lib.stride_tricks.as_strided
    shape = (outH, outW, kH, kW)
    strides = (A.strides[0],  # moving down one row in output
               A.strides[1],  # moving right one col in output
               A.strides[0],  # within a patch, moving down one row
               A.strides[1])  # within a patch, moving right one col
    patches = np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)
    
    # Now patches is shape (outH, outW, kH, kW).
    # We want to multiply each (kH,kW) patch by the kernel and sum over them.
    # "hwij,ij->hw" => (h,w,kH,kW), (kH,kW) => (h,w)
    out = np.einsum('hwij,ij->hw', patches, kernel)
    return out
