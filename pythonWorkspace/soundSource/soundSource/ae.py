#An auto encoder simply trains to mimic an input. neural connection from input to output. Nothing more. 

#arange 8x8 snippets from lena as 1x64 vector data. 
#train 100 features

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from theano import *
import theano.tensor as T
from theano import function

from skimage import data, color

image = data.lena()
grayImg = color.rgb2gray(image)

n_episodes = 1
w_samples = 8
n_samples = 100
n_hidden = 64

image_samples = np.ndarray((n_samples,w_samples,w_samples) )

#np.random.seed(1)

#Sample random cut-outs
for i in range(n_samples):
    rand_w = np.random.uniform(low=0.0, high=np.shape(grayImg)[0]-w_samples )
    rand_h = np.random.uniform(low=0.0, high=np.shape(grayImg)[1]-w_samples )
    image_samples[i] = grayImg[rand_w:rand_w+w_samples,rand_h:rand_h+w_samples]

#plot 20 first samples
for i in range(20):
    plt.subplot(4,5,i)
    plt.imshow(image_samples[i], cmap=cm.Greys_r, interpolation='nearest')

concImgSamps = np.ndarray((n_samples,w_samples*w_samples))

#Tested correct
for i in range(n_samples):
    for j in range(w_samples):
        for k in range(w_samples):
            concImgSamps[i][j*w_samples+k] = image_samples[i][j][k] 
    

#Whiten!

#Repeat
#Run through all samples
#for each node in hidden layer, calculate activation and output using :    
#calculate error
#backpropagate error derivative

#Declare input and bias vectors
x = np.ndarray(len(concImgSamps[0]))
b = np.ndarray(1)
b[0] = 1.00

n_in = len(x)
#this is AE, n_in = n_out
n_out = len(x)

#Declare weight matrices
W_ih = np.ndarray((n_in,n_hidden))
W_bh = np.ndarray((len(b),n_hidden))

W_ho = np.ndarray((n_hidden,n_out))
W_bo = np.ndarray((len(b),n_out))


#Initialize weigth matrices to random values to break symmetry
W_ih = np.random.uniform(low=-1.0, high=1.0, size=np.shape(W_ih))
W_bh = np.random.uniform(low=-1.0, high=1.0, size=np.shape(W_bh))

W_ho = np.random.uniform(low=-1.0, high=1.0, size=np.shape(W_ho))
W_bo = np.random.uniform(low=-1.0, high=1.0, size=np.shape(W_bo))


for i in range(n_episodes):
    e = 0.0
    for j in range(n_samples):
        x = concImgSamps[j]
        
        #Feed forward
        a_h = x.dot(W_ih) + b.dot(W_bh)
        y_h = 1/(1+np.exp(-a_h))
        
        a_o = y_h.dot(W_ho) + b.dot(W_bo)
        y_o = 1/(1+np.exp(-a_o))

        e += 0.5*np.power(np.linalg.norm(y_o-x, ord=2),2.0)

    W_sum = 0.0
    for j in range(W_ih[0]):
        for k in range(W_ih[1]):
            W_sum += np.power(W_ih[j][k],2.0)
    
    cost = e/double(n_samples) 

    #hip


#x = T.dvector('x')
#W = T.dmatrix('W')
#y = T.dvector('y')
#
#
#
#
#act = x*W
#sigm = 1/(1+T.exp(-act))
#sigmoid = function([act],sigm)
#
#rng = numpy.random
#
#N = 400
#feats = 784
#D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
#training_steps = 10000
#
## Declare Theano symbolic variables
#x = T.matrix("x")
#y = T.vector("y")
#w = theano.shared(rng.randn(feats), name="w")
#b = theano.shared(0., name="b")
#print "Initial model:"
#print w.get_value(), b.get_value()
#
## Construct Theano expression graph
#p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
#prediction = p_1 > 0.5                    # The prediction thresholded
#xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
#cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
#gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
#                                          # (we shall return to this in a
#                                          # following section of this tutorial)
#
## Compile
#train = theano.function(
#          inputs=[x,y],
#          outputs=[prediction, xent],
#          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
#predict = theano.function(inputs=[x], outputs=prediction)
#
## Train
#for i in range(training_steps):
#    pred, err = train(D[0], D[1])
#
#print "Final model:"
#print w.get_value(), b.get_value()
#print "target values for D:", D[1]
#print "prediction on D:", predict(D[0])





#print f([[1,2],[3,4]],[[10,20],[30,40]])