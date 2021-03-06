#An auto encoder simply trains to mimic an input. neural connection from input to output. Nothing more. 

#arange 8x8 snippets from lena as 1x64 vector data. 
#train 100 features

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from sklearn import decomposition

#from theano import *
#import theano.tensor as T
#from theano import function

from skimage import data, color

image = data.lena()
grayImg = color.rgb2gray(image)

n_epocs = 50
w_samples = 2
n_samples = 10000
n_hidden = 2
lr_alpha = 1.0
wd_lambda = 0.00
sp_rho = 0.05
sp_beta = 0.01

image_samples = np.ndarray((n_samples,w_samples,w_samples) )

#np.random.seed(1)

#Sample random cut-outs
for i in range(n_samples):
    rand_w = np.random.uniform(low=0.0, high=np.shape(grayImg)[0]-w_samples )
    rand_h = np.random.uniform(low=0.0, high=np.shape(grayImg)[1]-w_samples )
    image_samples[i] = grayImg[rand_w:rand_w+w_samples,rand_h:rand_h+w_samples]

##plot 20 first samples
#for i in range(20):
#    plt.subplot(4,5,i)
#    plt.imshow(image_samples[i], cmap=cm.Greys_r, interpolation='nearest')

concImgSamps = np.ndarray((n_samples,w_samples*w_samples))

#Tested correct
for i in range(n_samples):
    for j in range(w_samples):
        for k in range(w_samples):
            concImgSamps[i][j*w_samples+k] = image_samples[i][j][k] 
    

#Whiten!
pca = decomposition.PCA(whiten=True)
pca.fit(concImgSamps)

concImgSampsPCA = pca.transform(concImgSamps)


#Repeat
#Run through all samples
#for each node in hidden layer, calculate activation and output using :    
#calculate error
#backpropagate error derivative


#Declare input and bias vectors
x = np.ndarray(len(concImgSampsPCA[0]))
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

#based on suggested value from ufldl.stanford.edu
init_epsilon = 0.01#0.01

#Initialize weigth matrices to random values to break symmetry
W_ih = np.random.normal(loc=0.0, scale=init_epsilon**2.0, size=np.shape(W_ih))
W_bh = np.random.normal(loc=0.0, scale=init_epsilon**2.0, size=np.shape(W_bh))

W_ho = np.random.normal(loc=0.0, scale=init_epsilon**2.0, size=np.shape(W_ho))
W_bo = np.random.normal(loc=0.0, scale=init_epsilon**2.0, size=np.shape(W_bo))



e_list = []
W_ih_list = []

for i in range(n_epocs):
    print "epoc: " + str(i)
    e = 0.0
    
    delta_W_ih = np.zeros(np.shape(W_ih))
    delta_W_bh = np.zeros(np.shape(W_bh))
    delta_W_ho = np.zeros(np.shape(W_ho))
    delta_W_bo = np.zeros(np.shape(W_bo))

    a_h_mat = np.zeros((n_samples, n_hidden))
    a_o_mat = np.zeros((n_samples, n_out))

    for j in range(n_samples):
        x = concImgSampsPCA[j]
        y = x
        
        #Feed forward
        z_h = x.dot(W_ih) + b.dot(W_bh)
        a_h = 1.0/(1.0+np.exp(-z_h))

        a_h_mat[j] = a_h    
        
        z_o = a_h.dot(W_ho) + b.dot(W_bo)
        a_o = 1.0/(1.0+np.exp(-z_o))

        a_o_mat[j] = a_o

        e += 0.5*np.power(np.linalg.norm(a_o-y, ord=2),2.0)

    rho_hat = np.ndarray(n_hidden)
    
    for j in range(n_hidden):
        rho_hat_sum = 0.0
        for k in range(n_samples):
            rho_hat_sum = rho_hat_sum + a_h_mat[k,j]
        rho_hat[j] = rho_hat_sum/double(n_samples)  


    for j in range(n_samples):

        #Calculate error term for each layer, backpropagated through the weights        
        #'*' is element wise multiplocation
                
        e_delta_o = -(y-a_o_mat[j])*a_o_mat[j]*(1.0-a_o_mat[j])
        e_delta_h = (W_ho.dot(e_delta_o) + sp_beta*(-sp_rho/rho_hat + (1.0-sp_rho)/(1.0-rho_hat)))*a_h_mat[j]*(1.0-a_h_mat[j])
        
        #Calculate the gradient
        grad_W_ho = e_delta_o*a_o_mat[j]
        grad_W_bo = e_delta_o
        
        grad_W_ih = e_delta_h*a_h_mat[j]
        grad_W_bh = e_delta_h
        
        delta_W_ho = delta_W_ho + grad_W_ho
        delta_W_bo = delta_W_bo + grad_W_bo
        
        delta_W_ih = delta_W_ih + grad_W_ih
        delta_W_bh = delta_W_bh + grad_W_bh

        
        
        

    W_ho = W_ho - lr_alpha*(delta_W_ho/double(n_samples) + wd_lambda*W_ho)
    W_bo = W_bo - lr_alpha*delta_W_bo/double(n_samples)

    W_ih_list.append(W_ih[0,0])    
    
    W_ih = W_ih - lr_alpha*(delta_W_ih/double(n_samples) + wd_lambda*W_ih)
    W_bh = W_bh - lr_alpha*delta_W_bh/double(n_samples)


    W_sum = 0.0
    for j in range(np.shape(W_ih)[0]):
        for k in range(np.shape(W_ih)[1]):
            W_sum += np.power(W_ih[j][k],2.0)
    
    
    KL_div_sum = 0.0
    for j in range(n_hidden):
        KL_div_sum += sp_rho*np.log(sp_rho/rho_hat[j]) + (1.0-sp_rho)*np.log((1-sp_rho)/(1.0-rho_hat[j]))
        
    cost = e/double(n_samples) + wd_lambda*0.5*W_sum + sp_beta*KL_div_sum

    e_list.append(e)    


figure(1)
plt.plot(W_ih_list)



#REMEMBER TO TEST USING A SEPARATE TEST SET
#AND TRAIN USING A VALIDATION SET


figure(2)
plt.plot(e_list)
    #hip

feat_vis = np.ndarray((n_hidden,n_in))

for i in range(n_hidden):
    sum_W_ih = 0.0
    for j in range(n_in):
        sum_W_ih += W_ih[j,i]**2.0         
    for j in range(n_in):
        feat_vis[i,j] = W_ih[j,i]/np.sqrt(sum_W_ih)

#feat_vis_img = np.ndarray()

#transform hidden layer features based on pca features back to input space.
feat_vis_nonPCA = pca.inverse_transform(feat_vis)
feat_vis_img = feat_vis_nonPCA.reshape(((n_hidden,w_samples,w_samples)))



#plot the hidden features
figure(3) 
for i in range(n_hidden):
    plt.subplot(np.sqrt(n_hidden),np.sqrt(n_hidden),i)
    plt.imshow(feat_vis_img[i], cmap=cm.Greys_r, interpolation='nearest')


x = concImgSampsPCA[0]

z_h = x.dot(W_ih) + b.dot(W_bh)
a_h = 1.0/(1.0+np.exp(-z_h))

#a_h_mat[j] = a_h    

z_o = a_h.dot(W_ho) + b.dot(W_bo)
a_o = 1.0/(1.0+np.exp(-z_o))

#a_o_mat[j] = a_o

x_img = np.reshape(pca.inverse_transform(x),(np.sqrt(n_in),np.sqrt(n_in)))
a_o_img = np.reshape(pca.inverse_transform(a_o),(np.sqrt(n_in),np.sqrt(n_in)))

figure(4)
plt.subplot(121)
plt.imshow(x_img, cmap=cm.Greys_r, interpolation='nearest')
plt.subplot(122)
plt.imshow(a_o_img, cmap=cm.Greys_r, interpolation='nearest')

figure(5)
plt.plot(np.reshape(W_ih,(np.shape(W_ih)[0]*np.shape(W_ih)[1])))

#for i in range(n_hidden):
#    for j in range(w_samples):
#        for k in range(w_samples):
#            feat_vis_img[i,j,k] = feat_vis[i,j*w_samples+k]
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