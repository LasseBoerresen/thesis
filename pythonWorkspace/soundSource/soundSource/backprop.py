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

lr_alpha = 0.932423443
wd_lambda = 0.0001

sp_rho = 0.05
sp_beta = 0.01



def main():
    
    n_epocs = 5
    w_samples = 3
    n_samples = 100
    n_hidden = 4

#    image = data.lena()
#    grayImg = color.rgb2gray(image)
#
#    dat = dataExtractor(grayImg,n_samples,w_samples)


    x = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
    y = np.array([[0.0],[1.0],[1.0],[0.0]])

    nn = network((2,2,1), x, y)

#    res = nn.trainBFGS()
#    print res
    
    bestW = [ 33.72228738,  -4.82100946, -21.26940258,   4.62732186,
        14.67180313,   1.28878621, -47.76192686, -48.02730423,  54.34026228]
    
    nn.deFlattenWeights(bestW)    
    (res,gradCheck) = nn.train(n_epocs,x,y)
    
    print "gradCheck"
    print gradCheck[:,-4:-1]    
    
    print "res"
    print res



class network:
    def __init__(self, chain,x,y):
        self.chain = chain
        self.layers = []
        self.connections = []
        
        self.x = x
        self.y = y

        #Initiate layer objects in order        
        for i in range(len(self.chain)):        
            self.layers.append(layer(self.chain[i]))
            
        #Initiate connection objects in order
        for i in range(len(self.layers)-1):
            self.connections.append(connection(self.layers[i].m,self.layers[i+1].m))
            
    def trainBFGS(self):
        theta0 = self.flattenWeights()
   
        return minimize(self.cost, theta0, method='BFGS', jac=self.costJac, options={'disp': True}) #'gtol': 1e-6,

    def cost(self,theta):
        self.deFlattenWeights(theta)
        
        cost = 0.0
        for i in range(len(self.x)):
            self.feedForward(self.x[i])
            cost += 0.5*np.power(np.linalg.norm(self.layers[-1].a-self.y[i], ord=2),2.0)
        
        return cost

    def costJac(self,theta):
        self.deFlattenWeights(theta)
        #set accumulation matrices for weight vectors.  
        for i in range(len(self.connections)):
            self.connections[i].delta_W = np.zeros(np.shape(self.connections[i].delta_W))
            self.connections[i].delta_W_b = np.zeros(np.shape(self.connections[i].delta_W_b))

        #For each training example:
        for i in range(len(self.x)):
            self.feedForward(self.x[i]) #could complete entirely before feeding back, to be able to calcultate KL divergence.
            self.feedBack(self.y[i])       
        nabla = self.flattenGradients()/double(len(self.x))
 
        return nabla


    def train(self,epocs,x,y):
        res = np.ndarray((epocs,4))
        gradCheck = np.ndarray((epocs,len(self.flattenGradients())))
        
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
            (cost,gradCheck[i]) = self.backprop(x,y)
            
            #Update weight matrices.
            for i in range(len(self.connections)):
                self.connections[i].W = self.connections[i].W - lr_alpha*(self.connections[i].delta_W/double(len(x)) + wd_lambda*self.connections[i].W)
                self.connections[i].W_b = self.connections[i].W_b - lr_alpha*self.connections[i].delta_W_b/double(len(x))
              

        return (res,gradCheck)
    
    
    def backprop(self,x,y):
        
        #Reset accumulation matrices for weight vectors.  
        for i in range(len(self.connections)):
            self.connections[i].delta_W = np.zeros(np.shape(self.connections[i].delta_W))
            self.connections[i].delta_W_b = np.zeros(np.shape(self.connections[i].delta_W_b))

        cost = 0.0

        #For each training example:
        for i in range(len(x)):
            self.feedForward(x[i]) #could complete entirely before feeding back, to be able to calcultate KL divergence.
            self.feedBack(y[i])       

            cost += 0.5*np.power(np.linalg.norm(self.layers[-1].a-y[i], ord=2),2.0)
        
#        self.connections[0].delta_W
#        self.connections[0].delta_W_b

        
        ######################################################
        #gradient checking

        epsilon = 1e-4
#        nabla = self.flattenGradients()
        theta = self.flattenWeights()
        
        e = np.zeros(np.shape(theta))
        grad = np.zeros(np.shape(theta))
        
        #save gradients of actual weights.        
        nabla = self.flattenGradients()

        #For each weight individually, approximate gradient.
        for i in range(len(theta)):
            #Create modified theta vectors using e = [0,0,0,0,i=1,0,0].                
            e = np.zeros(np.shape(theta))
            e[i] = epsilon            
            thetaP = (theta + e)
            thetaN = (theta - e)

            costP = 0.0
            costN = 0.0
            
            #calculate sum of cost for entire dataset.
            self.deFlattenWeights(thetaP)             
            for j in range(len(x)):
                self.feedForward(x[j])
                #self.feedBack(y[j])
                costP += 0.5*np.power(np.linalg.norm(self.layers[-1].a-y[j], ord=2),2.0)
                
            self.deFlattenWeights(thetaN)                
            for j in range(len(x)):
                self.feedForward(x[j])
                #self.feedBack(y[i])                
                costN += 0.5*np.power(np.linalg.norm(self.layers[-1].a-y[j], ord=2),2.0)

            #calculate approximate gradient from consts
            grad[i] = ( costP/double(len(x)) - costN/double(len(x)) )/(2.0*epsilon)             
            
        #####################################################
            
#        #Update weight matrices.
#        for i in range(len(self.connections)):
#            self.connections[i].W = self.connections[i].W - lr_alpha*(self.connections[i].delta_W/double(len(x)) + wd_lambda*self.connections[i].W)
#            self.connections[i].W_b = self.connections[i].W_b - lr_alpha*self.connections[i].delta_W_b/double(len(x))
#           
        return (cost,np.abs(nabla-grad))
        
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
        
        #Calculate weights updates for each connection, based on this trainins example
        for i in range(len(self.connections)):
            #Calculate gradients from activations and the errors of the following layer.
            grad_W = np.outer(self.layers[i].a,self.layers[i+1].err) #np.outer is used to force treating the vectors as matrices 9x1 * 1x9 = 9x9
            grad_W_b = self.layers[i+1].err
            
            #Calc
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

    def flattenWeights(self):
        theta = np.array([])

        for i in range(len(self.connections)):
            W_shape = np.shape(self.connections[i].W)
            W_b_shape = np.shape(self.connections[i].W_b)
            theta = np.concatenate((theta, self.connections[i].W.reshape((W_shape[0]*W_shape[1])), self.connections[i].W_b.reshape((W_b_shape[0]*W_b_shape[1])) ))
        
        return theta
    
    def flattenGradients(self):   
        nabla = np.array([])

        for i in range(len(self.connections)):
            delta_W_shape = np.shape(self.connections[i].delta_W)
            delta_W_b_shape = np.shape(self.connections[i].delta_W_b)
            nabla = np.concatenate((nabla, self.connections[i].delta_W.reshape((delta_W_shape[0]*delta_W_shape[1])), self.connections[i].delta_W_b.reshape((delta_W_b_shape[0]*delta_W_b_shape[1])) ))
        
        return nabla
    
    #OBS CHECK THIS
    def deFlattenWeights(self,theta):
        count = 0
        for i in range(len(self.connections)):
            W_shape = np.shape(self.connections[i].W)
            W_b_shape = np.shape(self.connections[i].W_b)
            
            self.connections[i].W = np.reshape(theta[count:count+W_shape[0]*W_shape[1]], (W_shape[0],W_shape[1]))
            count += W_shape[0]*W_shape[1]
            self.connections[i].W_b = np.reshape(theta[count:count+W_b_shape[0]*W_b_shape[1]], (W_b_shape[0],W_b_shape[1]))
            count += W_b_shape[0]*W_b_shape[1]
            
       

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






if __name__ == '__main__':
    main()