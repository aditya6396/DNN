#!/usr/bin/env python
# coding: utf-8

# # Assignment 1 –Lab:
# o Implement the main steps of a Shallow Neural Network • Understand the dataset • Implement your first Forward and Backward propagation • Implement activation function, gradient descent • Build Neural Network Model • Test and optimize the model • Make Predictions (to be done in Jupyter notebook. Need Jupyter-notebook and libraries installed)

# In[ ]:


Step 1 : Creating the data set using numpy array of 0s and 1s. 
As the image is a collection of pixel values in matrix, we will create those matrix of pixel for A, B, C 

using 0 and 1
#A
0 0 1 1 0 0
0 1 0 0 1 0 
1 1 1 1 1 1
1 0 0 0 0 1 
1 0 0 0 0 1

#B
0 1 1 1 1 0
0 1 0 0 1 0 
0 1 1 1 1 0
0 1 0 0 1 0
0 1 1 1 1 0

#C
0 1 1 1 1 0
0 1 0 0 0 0
0 1 0 0 0 0
0 1 0 0 0 0
0 1 1 1 1 0

#Labels for each Letter
A=[1, 0, 0]
B=[0, 1, 0]
C=[0, 0, 1]


# In[15]:


#Step 1 : Creating the data set using numpy array of 0s and 1s. 
#As the image is a collection of pixel values in matrix, we will create those matrix of pixel for A, B, C 


 
# A
a =0, 0, 1, 1, 0, 0,
   0, 1, 0, 0, 1, 0,
   1, 1, 1, 1, 1, 1,
   1, 0, 0, 0, 0, 1,
   1, 0, 0, 0, 0, 1
# B
b =0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0
# C
c =0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 1, 1, 1, 0
 
# Creating labels
y =[1, 0, 0],
   [0, 1, 0],
   [0, 0, 1]


# In[17]:


# converting data and labels into numpy array
 
"""
Convert the matrix of 0 and 1 into one hot vector
so that we can directly feed it to the neural network,
these vectors are then stored in a list x.
"""
 
x =[np.array(a).reshape(1, 30), np.array(b).reshape(1, 30),
                                np.array(c).reshape(1, 30)]
 
 
# Labels are also converted into NumPy array
y = np.array(y)
 
 
print(x, "\n\n", y)


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
# visualizing the data, ploting A.
plt.imshow(np.array(a).reshape(5, 6))
plt.show()


# In[ ]:





# Step 4 : Defining the architecture or structure of the deep neural network. This includes deciding the number of layers and the number of nodes in each layer. Our neural network is going to have the following structure.
# 
# 1st layer: Input layer(1, 30) 2nd layer: Hidden layer (1, 5) 3rd layer: Output layer(3, 3)
# ![image.png](attachment:image.png)
# 
# 

# ### Step 5: Declaring and defining all the function to build deep neural network.

# In[19]:


# activation function
 
def sigmoid(x):
    return(1/(1 + np.exp(-x)))
   
# Creating the Feed forward neural network
# 1 Input layer(1, 30)
# 1 hidden layer (1, 5)
# 1 output layer(3, 3)
 
def f_forward(x, w1, w2):
    # hidden
    z1 = x.dot(w1)# input from layer 1
    a1 = sigmoid(z1)# out put of layer 2
     
    # Output layer
    z2 = a1.dot(w2)# input of out layer
    a2 = sigmoid(z2)# output of out layer
    return(a2)
  
# initializing the weights randomly
def generate_wt(x, y):
    l =[]
    for i in range(x * y):
        l.append(np.random.randn())
    return(np.array(l).reshape(x, y))
     
# for loss we will be using mean square error(MSE)
def loss(out, Y):
    s =(np.square(out-Y))
    s = np.sum(s)/len(y)
    return(s)
   
# Back propagation of error
def back_prop(x, y, w1, w2, alpha):
     
    # hidden layer
    z1 = x.dot(w1)# input from layer 1
    a1 = sigmoid(z1)# output of layer 2
     
    # Output layer
    z2 = a1.dot(w2)# input of out layer
    a2 = sigmoid(z2)# output of out layer
    
    # error in output layer
    d2 =(a2-y)
    d1 = np.multiply((w2.dot((d2.transpose()))).transpose(),
                                   (np.multiply(a1, 1-a1)))
 
    # Gradient for w1 and w2
    w1_adj = x.transpose().dot(d1)
    w2_adj = a1.transpose().dot(d2)
     
    # Updating parameters
    w1 = w1-(alpha*(w1_adj))
    w2 = w2-(alpha*(w2_adj))
     
    return(w1, w2)
 
def train(x, Y, w1, w2, alpha = 0.01, epoch = 10):
    acc =[]
    losss =[]
    for j in range(epoch):
        l =[]
        for i in range(len(x)):
            out = f_forward(x[i], w1, w2)
            l.append((loss(out, Y[i])))
            w1, w2 = back_prop(x[i], y[i], w1, w2, alpha)
        print("epochs:", j + 1, "======== acc:", (1-(sum(l)/len(x)))*100)  
        acc.append((1-(sum(l)/len(x)))*100)
        losss.append(sum(l)/len(x))
    return(acc, losss, w1, w2)
  
def predict(x, w1, w2):
    Out = f_forward(x, w1, w2)
    maxm = 0
    k = 0
    for i in range(len(Out[0])):
        if(maxm<Out[0][i]):
            maxm = Out[0][i]
            k = i
    if(k == 0):
        print("Image is of letter A.")
    elif(k == 1):
        print("Image is of letter B.")
    else:
        print("Image is of letter C.")
    plt.imshow(x.reshape(5, 6))
    plt.show()   
   


# In[20]:


w1 = generate_wt(30, 5)
w2 = generate_wt(5, 3)
print(w1, "\n\n", w2)


# ### Step 7 : Training the model.

# In[21]:


"""The arguments of train function are data set list x,
correct labels y, weights w1, w2, learning rate = 0.1,
no of epochs or iteration.The function will return the
matrix of accuracy and loss and also the matrix of
trained weights w1, w2"""
 
acc, losss, w1, w2 = train(x, y, w1, w2, 0.1, 100)


# ### Step 8 : Plotting the graphs of loss and accuracy with respect to number of epochs(Iteration).

# In[23]:


import matplotlib.pyplot as plt1
 
# ploting accuracy
plt1.plot(acc)
plt1.ylabel('Accuracy')
plt1.xlabel("Epochs:")
plt1.show()
 
# plotting Loss
plt1.plot(losss)
plt1.ylabel('Loss')
plt1.xlabel("Epochs:")
plt1.show()


# In[24]:


# the trained weights are
print(w1, "\n", w2)


# In[25]:


"""
The predict function will take the following arguments:
1) image matrix
2) w1 trained weights
3) w2 trained weights
"""
predict(x[1], w1, w2)


# # Assignment 2 –Lab:
# 

# ### Sigmoid Function
# The sigmoid function is a special form of the logistic function and is usually denoted by σ(x) or sig(x). It is given by:
# 
# σ(x) = 1/(1+exp(-x))

# ## Plotting Sigmoid 2D 

# In[2]:


import matplotlib.pyplot as plt
from scipy.special import expit
import numpy as np

#define range of x-values
x = np.linspace(-10, 10, 100)

#calculate sigmoid function for each x-value
y = expit(x)
  
#create plot
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('F(x)')

#display plot
plt.show()


# ## Plotting Sigmoid 3D

# In[8]:


def sigmoid_2d(x1, x2, w1, w2, b):
  return 1/(1 + np.exp(-(w1*x1 + w2*x2 + b)))


# In[9]:


sigmoid_2d(1, 0, 0.5, 0, 0)


# In[10]:


X1 = np.linspace(-4, 5, 50)
X2 = np.linspace(-4, 5, 50)

XX1, XX2 = np.meshgrid(X1, X2)

print(X1.shape, X2.shape, XX1.shape, XX2.shape)


# In[11]:


w1 = 2
w2 = -0.5
b = 1
Y = sigmoid_2d(XX1, XX2, w1, w2, b)


# In[24]:


fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.plot_surface(XX1, XX2, Y, cmap='PiYG')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')

ax.view_init(30, 270)


# In[ ]:





# ## Contour Plot 
# A contour plot is a graphical method to visualize the 3-D surface by plotting constant Z slices called contours in a 2-D format.
# The contour plot is an alternative to a 3-D surface plotA contour plot is a graphical method to visualize the 3-D surface by plotting constant Z slices called contours in a 2-D format.
# The contour plot is an alternative to a 3-D surface plot

# In[29]:


import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-1.0, 1.0, 100)
ylist = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = np.sqrt(X**2 + Y**2)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
#ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.show()


# ## Plotting Loss

# In[30]:


x = np.arange(0, 1, 0.01)
y = -np.log(x)

plt.plot(x, y, color='k', lw=1, linestyle=None)
plt.show()


# ## Standardization 

# Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks.

# ## Test/Train split

# In[35]:


import keras
from tensorflow.keras.datasets.mnist import load_data 
from tensorflow.keras.datasets import mnist 


# In[37]:


#loading the MNIST Dataset 

(X_train,y_train), (X_test,Y_test) = mnist.load_data()
train_size = X_train.shape[0]
batch_size = 3
batch_mask = np.random.choice(train_size, batch_size)
print(X_train[batch_mask])
print(y_train[batch_mask])


# # Assignment –Lab 3:¶

# ## Implement a two-class neural network with a hidden layer , Implement forward and backward propagation

# ## Import packages

# In[59]:


import keras

from tensorflow.keras.datasets.mnist import load_data 
from tensorflow.keras.datasets import mnist 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras as tk
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from keras.layers.core import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential


# In[61]:


#loading the MNIST Dataset 

(X_train,y_train), (X_test,Y_test) = mnist.load_data()
X_train.shape
y_train.shape


# In[77]:


#printing the shapes 

print(X_train.shape, y_train.shape)

print(X_test.shape , y_test.shape)


# In[78]:


#reshaping train and test sets 

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

X_test = X_test .reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))


# In[80]:


#printing the shapes 

print(X_train.shape, y_train.shape)

print(X_test.shape , y_test.shape)

 


# In[ ]:





# In[90]:


#Normaliziation
#normalizing the pixel values of images 

X_train = X_train.astype('float32')/255.0

X_test = X_test.astype('float32')/255.0


# In[ ]:





# ## Let’s first write multi-layer model function to implement gradient-based learning using predefined number of iterations and learning rate.
# 
# 

# In[82]:


model42 = keras.models.Sequential([
                         keras.layers.Flatten(input_shape=[28,28]),
                         keras.layers.Dense(units=256, activation='sigmoid'),
                         keras.layers.Dense(units=128, activation='relu'),
    
                         keras.layers.Dense(units=128, activation='relu'),
                         
                         keras.layers.Dense(units=10, activation='softmax'),
])


# In[83]:


model42.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[84]:


history=model42.fit(X_train,y_train,epochs=10,validation_split=0.20, 
    batch_size=20, 
    verbose=2)


# ## Assignment –Lab 4:

o Implement Regularization,

o Dropouts 

o Improve performance of the learning algorithms

# In[58]:


import keras

from tensorflow.keras.datasets.mnist import load_data 
from tensorflow.keras.datasets import mnist 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras as tk
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from keras.layers.core import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential


# # L2 

# In[56]:





# In[ ]:


model46 = keras.models.Sequential([
                         keras.layers.Flatten(input_shape=[28,28]),
                         keras.layers.Dense(units=256, activation='relu',kernel_regularizer=regularizers.l2(0.0001)),
                         keras.layers.Dense(units=128, activation='relu'),
                         keras.layers.Dense(units=10, activation='softmax'),
])


# In[57]:


model46.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[58]:


history1=model46.fit(X_train,y_train,epochs=10,validation_split=0.20, 
    batch_size=20, 
    verbose=2)


# ## L1

# In[93]:


model47 = keras.models.Sequential([
                         keras.layers.Flatten(input_shape=[28,28]),
                         keras.layers.Dense(units=256, activation='relu',kernel_regularizer=regularizers.l2(0.0001)),
                         keras.layers.Dense(units=128, activation='relu'),
                         keras.layers.Dense(units=10, activation='softmax'),
])


# In[63]:


model47.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[64]:


history2=model47.fit(X_train,y_train,epochs=10,validation_split=0.20, 
    batch_size=20, 
    verbose=2)


# ## Dropout

# In[95]:


model45 = keras.models.Sequential([
                         keras.layers.Flatten(input_shape=[28,28]),
                         keras.layers.Dense(units=256, activation='relu',kernel_regularizer=regularizers.l2(0.0001)),
                         keras.layers.Dense(units=128, activation='relu'),
                         keras.layers.Dense(units=10, activation='softmax'),
])


# In[96]:


model45.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[97]:


history4=model45.fit(X_train,y_train,epochs=10,validation_split=0.20, 
    batch_size=20, 
    verbose=2)


# In[71]:



acc = history4.history['accuracy']
val_acc = history4.history['val_accuracy']
loss = history4.history['loss']
val_loss = history4.history['val_loss']
epochs_range = range(len(acc))
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# By using Droupout as Regularization method
# 
# Improve performance of the learning algorithms
# 
#  

# In[ ]:




