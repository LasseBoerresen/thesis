#this is a reboot of the neural net prototype. It had gotten to complicated, with several layers too fast. 
#All I want is a working neural network with backpropagation that calculates the derivative correctly. 

#TODO:I should work towards being able to use a general solver. 

#TODO: test on mnist dataset, classifying digits

#How can I do it as an object oriented system. 

#what about data?

#Each Layer could be an object. Also each connection. 

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from sklearn import decomposition

from skimage import data, color

from scipy.optimize import minimize, rosen, rosen_der

lr_alpha = 13.32423443
wd_lambda = 0.0
sp_rho = 0.05
sp_beta = 0.01


class network:
    def __init__(self, chain):
        self.chain = chain
        self.layers = []
        self.connections = []

        #Initiate layer objects in order        
        for i in range(len(self.chain)):        
            self.layers.append(layer(self.chain[i]))
            
        #Initiate connection objects in order
        for i in range(len(self.layers)-1):
            self.connections.append(connection(self.layers[i].m,self.layers[i+1].m))
            
        
    def train(self,epocs,x,y):
        res = np.ndarray((epocs,4))
        
        for i in range(epocs):
            resTemp = np.ndarray((4))            
            self.feedForward(np.array([0.0,0.0]))
            resTemp[0] = self.layers[-1].a
            self.feedForward(np.array([0.0,1.0]))
            resTemp[1] = self.layers[-1].a
            self.feedForward(np.array([1.0,0.0]))
            resTemp[2] = self.layers[-1].a
            self.feedForward(np.array([1.0,1.0]))
            resTemp[3] = self.layers[-1].a
            res[i] = resTemp
            
            #print self.connections[-1].W
#            print self.connections[-1].W_b
            self.backprop(x,y)

        return res
        
    def backprop(self,x,y):
        
        #Reset accumulation matrices for weight vectors.  
        for i in range(len(self.connections)):
            self.connections[i].delta_W = np.zeros(np.shape(self.connections[i].delta_W))
            self.connections[i].delta_W_b = np.zeros(np.shape(self.connections[i].delta_W_b))

        #For each training example:
        for i in range(len(x)):
            self.feedForward(x[i]) #could complete entirely before feeding back, to be able to calcultate KL divergence.
            self.feedBack(y[i])        
        
        np.        
        
        #Update weight matrices.
        for i in range(len(self.connections)):
            self.connections[i].W = self.connections[i].W - lr_alpha*(self.connections[i].delta_W/double(len(x)) + wd_lambda*self.connections[i].W)
            self.connections[i].W_b = self.connections[i].W_b - lr_alpha*self.connections[i].delta_W_b/double(len(x))
           
        
    #feed ONE sample forward through
    def feedForward(self, x):
#        x = concImgSamps[j]
#        y = x
#       
        #Feed values forward for each layer sequentially
        self.layers[0].a = x 
        for i in range(len(self.layers)-1):
            self.layers[i+1].z = self.layers[i].a.dot(self.connections[i].W) + self.layers[i].b.dot(self.connections[i].W_b)            
            self.layers[i+1].a = 1.0/(1.0+np.exp(-self.layers[i+1].z))


#
#        #Feed forward
#        z_h = x.dot(W_ih) + b.dot(W_bh)
#        a_h = 1.0/(1.0+np.exp(-z_h))
#
#        a_h_mat[j] = a_h    
#        
#        z_o = a_h.dot(W_ho) + b.dot(W_bo)
#        a_o = 1.0/(1.0+np.exp(-z_o))
#
#        a_o_mat[j] = a_o
#
#        e += 0.5*np.power(np.linalg.norm(a_o-y, ord=2),2.0)

    def feedBack(self,y):
        #find error for output layer        
        self.layers[-1].err = -(y-self.layers[-1].a)*self.layers[-1].a*(1.0-self.layers[-1].a)
        
        #go through all hidden layers, i.e. not last and first layer, hence -1 and .pop(0)
        seq = range(len(self.layers)-1)
        seq.pop(0)        
        for i in reversed(seq):
            self.layers[i].err = (self.connections[i].W.dot(self.layers[i+1].err))*self.layers[i].a*(1.0-self.layers[i].a)                      
        
        #TODO: Do numerical check of gradients.
        for i in range(len(self.connections)):
            grad_W = np.outer(self.layers[i].a,self.layers[i+1].err) #np.outer is used to force treating the vectors as matrices 9x1 * 1x9 = 9x9
            grad_W_b = self.layers[i+1].err
            
            self.connections[i].delta_W = self.connections[i].delta_W + grad_W
            self.connections[i].delta_W_b = self.connections[i].delta_W_b + grad_W_b 

                    
        
#        #Calculate error term for each layer, backpropagated through the weights        
#        #'*' is element wise multiplocation
#                
#        e_delta_o = -(concImgSamps[j]-a_o_mat[j])*a_o_mat[j]*(1.0-a_o_mat[j])
##        e_delta_h = (W_ho.dot(e_delta_o) + sp_beta*(-sp_rho/rho_hat + (1-sp_rho)/(1-rho_hat)))*a_h_mat[j]*(1.0-a_h_mat[j])
#        e_delta_h = (W_ho.dot(e_delta_o))*a_h_mat[j]*(1.0-a_h_mat[j])
#
#        
#        #Calculate the gradient
#        grad_W_ho = np.outer(a_h_mat[j],e_delta_o) #np.outer is used to force treating the vectors as matrices 9x1 * 1x9 = 9x9
#        grad_W_bo = e_delta_o
#    
#        grad_W_ih = np.outer(concImgSamps[j],e_delta_h)
#        grad_W_bh = e_delta_h
#        
#        delta_W_ho = delta_W_ho + grad_W_ho
#        delta_W_bo = delta_W_bo + grad_W_bo
#        
#        
#        delta_W_ih = delta_W_ih + grad_W_ih
#        delta_W_bh = delta_W_bh + grad_W_bh

        

class layer:
    def __init__(self,m):
       self.m = m
       self.b = np.array([1.0])
       self.z = np.ndarray((m))
       self.a = np.ndarray((m))

       
       self.err = np.ndarray((m)) 

    
class connection:
    def __init__(self, nIn, nOut):
        self.W = np.ndarray((nIn,nOut))
        self.W_b = np.ndarray((1,nOut))
        
        self.delta_W = np.zeros((nIn,nOut))
        self.delta_W_b = np.zeros((1,nOut))
        
        init_epsilon = 1.01#0.01 #going from 0.01 to 1.01 made it possible to train an xor function. Before, with all values at approx 0.00001, all outputs just went towards 0.5, i.e. avrg output.
        #Initialize weigth matrices to random values to break symmetry
        self.W = np.random.normal(loc=0.0, scale=init_epsilon**2.0, size=np.shape(self.W))
        self.W_b = np.random.normal(loc=0.0, scale=init_epsilon**2.0, size=np.shape(self.W_b))



class dataExtractor:
    def __init__(self,img,n_samples,w_samples):
        self.image_samples = np.ndarray((n_samples,w_samples,w_samples) )
        
        #Sample random cut-outs
        for i in range(n_samples):
            rand_w = np.random.uniform(low=0.0, high=np.shape(img)[0]-w_samples )
            rand_h = np.random.uniform(low=0.0, high=np.shape(img)[1]-w_samples )
            self.image_samples[i] = img[rand_w:rand_w+w_samples,rand_h:rand_h+w_samples]
        
        ##plot 20 first samples
        #for i in range(20):
        #    plt.subplot(4,5,i)
        #    plt.imshow(image_samples[i], cmap=cm.Greys_r, interpolation='nearest')
        
        self.concImgSamps = np.ndarray((n_samples,w_samples*w_samples))
        
        #Tested correct
        for i in range(n_samples):
            for j in range(w_samples):
                for k in range(w_samples):
                    self.concImgSamps[i][j*w_samples+k] = image_samples[i][j][k] 
            
        #TODO: REMEMBER TO NORMALIZE THE DATA
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            
        #Whiten!
        pca = decomposition.PCA(whiten=True)
        pca.fit(self.concImgSamps)
        
        #OBS: REMEMBER TO TRANSFORM BACK WHEN EVALUATING FEATURES
        self.concImgSampsPCA = pca.transform(self.concImgSamps)



def main():
    
    n_epocs = 1000
    w_samples = 3
    n_samples = 100
    n_hidden = 4

#    image = data.lena()
#    grayImg = color.rgb2gray(image)
#
#    dat = dataExtractor(grayImg,n_samples,w_samples)

    nn = network((2,2,1))
    
    print "nn.connections[0].W_b"    
    print nn.layers[0].z 
    print nn.layers[0].a 
    
    print
    print
    
    x = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
    y = np.array([[0.0],[1.0],[1.0],[0.0]])

    nn.feedForward(np.array([0.0,1.0]))
    print nn.layers[0].a
    res = nn.train(n_epocs,x,y)
    
    print res



if __name__ == '__main__':
    main()