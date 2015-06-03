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

from sklearn.preprocessing import normalize


#henrikc = sp.misc.imread("henrik1.png")
##henrik = sp.misc.lena()
#plt.subplot(2,1,1)
#plt.imshow(henrikc, cmap=cm.Greys_r, interpolation='nearest', vmin=0, vmax=255)
#
#plt.subplot(2,1,2)
#
#henrik = color.rgb2gray(henrikc)
#plt.imshow(henrik, cmap=cm.Greys_r, interpolation='nearest', vmin=0, vmax=255)
#print henrik
#henrik = henrik.flatten()
#
#print "henrik.shape"
#print henrik.shape
#
#
#xMin = np.min(henrik.flatten())
#xMax = np.max(henrik.flatten())
#henrikNorm = (henrik-xMin)/(xMax-xMin)

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')#, data_home=custom_data_home)

mnistBinData = []
mnistBinTarget = []
for i in range(len(mnist.target)):
    if mnist.target[i] < 2:
        mnistBinData.append(mnist.data[i])        
        mnistBinTarget.append(mnist.target[i])
mnistBinData = np.array(mnistBinData)
mnistBinTarget = np.array(mnistBinTarget)

mnistBinTarget1Hot = np.zeros((mnistBinTarget.shape[0],2))
for i in range(len(mnistBinTarget)):
    for j in np.unique(mnistBinTarget):
        if mnistBinTarget[i] == j:
            mnistBinTarget1Hot[i,j] = 1.0 


K = 10 #Classes

unique = np.unique(mnist.target)

mnistTarget1Hot = np.zeros((len(mnist.target),K))
for i in range(len(mnist.target)):
    for j in unique:
        if mnist.target[i] == j:
            mnistTarget1Hot[i,j] = 1.0



#print "np.unique(mnist.target)"
#print mnistBinData.shape
#firstImg = np.reshape(mnistBinData[0],(28,28))
#print firstImg.shape
#plt.imshow(firstImg, cmap=cm.Greys_r, interpolation='nearest', vmin=0, vmax=255)
#



lr_alpha = 0.932423443
wd_lambda = 0.00001

sp_rho = 0.05
sp_beta = 0.01



def main():
    
    n_epocs = 0
    w_samples = 3
    n_samples = 5000
    n_hidden = 4

#    image = data.lena()
#    grayImg = color.rgb2gray(image)
#
#    dat = dataExtractor(grayImg,n_samples,w_samples)


#    x = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
#    y = np.array([[0.0],[0.0],[0.0],[1.0]])
    
#    xNorm = normalize(mnistBinData.astype(double))

    #NORMALIZE DATA, i.e. Scale features to lie between 0.0 and 1.0   
    
    
    x = mnist.data.astype(double)
#    x = np.concatenate((mnistBinData.astype(double)[0:n_samples/2],mnistBinData.astype(double)[-(n_samples/2+1):-1]))
#    x = mnistBinData.astype(double) 
    xMin = np.min(x.flatten())
    xMax = np.max(x.flatten())
    xNorm = (x-xMin)/(xMax-xMin)
    
    #ERROR: ONLY RETURNS n_conponents features, equal to n samples, instead of all features.
    #Run PCA on data, to decorrelate features.
#    pca = decomposition.PCA(whiten=True)
#    pca.fit(xNorm)
#    xNormPca = pca.transform(xNorm)
    
    xNormPca = xNorm #CHEAT   
    
    x = xNormPca
    y = mnistTarget1Hot    
#    y = mnistBinTarget1Hot#np.concatenate((mnistBinTarget1Hot[0:n_samples/2],mnistBinTarget1Hot[-(n_samples/2+1):-1])) #mnistTarget1Hot


    #Shuffle Dataset
    oldx = x
    oldy = y
    order = np.arange(len(x))
    np.random.shuffle(order)
    for i in range(len(x)):
        x[i] = oldx[order[i]]
        y[i] = oldy[order[i]]    
       
    #Extract n_samples.        
    x = x[0:n_samples]
    y = y[0:n_samples]    
    
    print "x.shape:"
    print x.shape
    print
    print "y.shape:"
    print y.shape
#    print y
    print
    
    print
    print "Data Ready "
    print
    
    nn = network((len(x[0]),255,len(y[0])), x, y)

    #Load previous weights    
    nn.deFlattenWeights(sp.load("bestW.npy"))
    print
    print "Network Created"
    print

#    print henrikNorm.shape
#    nn.feedForward(henrikNorm)
#    print "henrik result:"
#    print nn.layers[-1].a

#
#    res = nn.trainBFGS()
#    bestW = res.x
#    sp.save("bestW.npy",bestW)
#
#    print
#    print "Network Trained"
#    print
#    print res

#    
#    nn.deFlattenWeights(bestW)    
#    (res,gradCheck) = nn.train(n_epocs,x,y)
    
#    print "gradCheck"
#    print gradCheck[:,-4:-1]    
#    
#    print "res"
#    print res
#    
    confMat = np.zeros((len(y[0]),len(y[0])),dtype = int)

    nCorrect= 0
    for i in range(len(x)):
        nn.feedForward(x[i])
        confMat[y[i].argmax(),nn.layers[-1].a.argmax()] += 1
        
    print
    print "confMat"
    print confMat
    print
    
    for i in range(len(confMat)):
        nCorrect += confMat[i,i]
    print "%correct"
    print nCorrect/double(len(x))


    


#    #calculate optimal images for each feature.
#    cons = {'type': 'ineq', 'fun': lambda x: -(np.power(np.linalg.norm(x,ord=2),2.0)-1.0)}
#    bounds = []
#    for i in range(len(x[0])):
#        bounds.append((0.,1.))
##    plt.figure(0)  
##    plt.imshow(np.reshape(x[0],(28,28)), cmap=cm.Greys_r, interpolation='nearest', vmin=0., vmax=1.)
#
#    featImgs = np.ndarray((len(nn.layers[-1].a),28,28))
#    for i in range(len(nn.layers[-1].a)):
#        nn.feat = i
##        res = minimize(nn.featCost, np.random.uniform(low=0.0, high=0.3,size=(28*28)), method='SLSQP', bounds=bounds, constraints=cons, options={'disp': True})
#        res = minimize(nn.featCost, np.random.uniform(low=0.0, high=0.3,size=(28*28)), method='SLSQP', bounds=bounds, options={'disp': True})
#
#        featImgs[i] = np.reshape(res.x,(28,28))
#
#    sp.save("featImgs.npy",featImgs)


    maxed = np.ndarray((len(nn.layers[1].a),len(nn.layers[0].a)))
    #for number of hidden units.
    for i in range(len(nn.layers[1].a)):
        #for each input/pixel        
        normW = np.linalg.norm(nn.connections[0].W[:,i])
        for j in range(len(nn.layers[0].a)):
            maxed[i,j] = nn.connections[0].W[j,i]/normW

    print np.linalg.norm(maxed[0])
    print maxed[0]
    plt.figure(1)
    for i in range(100):
        plt.subplot(10,10,i)
        plt.imshow(np.reshape(maxed[i],(28,28)), cmap=cm.Greys_r, interpolation='nearest')#, vmin=0., vmax=1.)


#    #For each class
#    for i in range(len(nn.layers[-1].a)):
#        #for each pixel/input        
#        for i in range(len(nn.layers[0].a)):
#            

#    featImgs = sp.load("featImgs.npy")
#
#    for i in range(len(featImgs)):    
#        nn.feedForward(np.reshape(featImgs[i],(28*28)))
#        print
#        print "i:" + str(i)
#        print nn.layers[-1].a[i]
#        print
#        print -(np.power(np.linalg.norm(np.reshape(featImgs[i],(28*28)),ord=2),2.0)-1.0)

#    print featImgs[0].flatten()
    

class network:
    def __init__(self, chain,x,y):
        self.chain = chain
        self.layers = []
        self.connections = []
        
        self.x = x
        self.y = y
        
        self.feat = 0

        #Initiate layer objects in order        
        for i in range(len(self.chain)):        
            self.layers.append(layer(self.chain[i]))
            
        #Initiate connection objects in order
        for i in range(len(self.layers)-1):
            self.connections.append(connection(self.layers[i].m,self.layers[i+1].m))

    def featCost(self,theta):
        self.feedForward(theta)
        cost = -(self.layers[-1].a[self.feat] - 1.0*(np.power(np.linalg.norm(theta,ord=2),2.0)))
#        print cost        
        return cost
        
 
            
    def trainBFGS(self):
        theta0 = self.flattenWeights()
   
        return minimize(self.cost, theta0, method='CG', jac=self.costJac, options={'maxiter' : 10,'gtol': 1e-8,'disp': True}) #

    

    def cost(self,theta):
        self.deFlattenWeights(theta)
        
        cost = 0.0
        for i in range(len(self.x)):
            self.feedForward(self.x[i])
            cost += 0.5*np.power(np.linalg.norm(self.layers[-1].a-self.y[i], ord=2),2.0)

        weightSum =  np.sum(np.power(theta,2.0))
        
        finalCost = cost/double(len(self.x)) + wd_lambda/2.0*weightSum
#        print "cost:"        
        print finalCost
        return finalCost
        
    def costJac(self,theta):
        self.deFlattenWeights(theta)
        #set accumulation matrices for weight vectors to 0.0.  
        for i in range(len(self.connections)):
            self.connections[i].delta_W = np.zeros(np.shape(self.connections[i].delta_W))
            self.connections[i].delta_W_b = np.zeros(np.shape(self.connections[i].delta_W_b))

        #For each training example:
        for i in range(len(self.x)):
            self.feedForward(self.x[i]) #could complete entirely before feeding back, to be able to calcultate KL divergence.
            self.feedBack(self.y[i])       
        nabla = self.flattenGradients()/double(len(self.x))
#        print "nabla:"
#        print np.linalg.norm(nabla,ord=2)        
        return nabla


    def costLog(self,theta):
        self.deFlattenWeights(theta)
        
        cost = 0.0
        for i in range(len(self.x)):
            self.feedForward(self.x[i])
            cost += self.y[i]*(np.log(layers[-1].a)) + (1.0-self.y[i])*(np.log(layers[-1].a))

        weightSum =  np.sum(np.power(theta,2.0))
        
        return (-cost/double(len(self.x))) + wd_lambda/2.0*weightSum

    def costLogJac(self,theta):
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
        res = np.ndarray((epocs,4,2))
        gradCheck = np.ndarray((epocs,len(self.flattenGradients())))
        
        for i in range(epocs):
#            resTemp = np.ndarray((4,2))            
#            self.feedForward(np.array(x[0]))
#            resTemp[0] = self.layers[-1].a
#            self.feedForward(np.array(x[1]))
#            resTemp[1] = self.layers[-1].a
#            self.feedForward(np.array(x[-1]))
#            resTemp[2] = self.layers[-1].a
#            self.feedForward(np.array(x[-2]))
#            resTemp[3] = self.layers[-1].a
#            res[i] = resTemp
            
            #print self.connections[-1].W
#            print self.connections[-1].W_b
            (cost,gradCheck[i]) = self.backprop(x,y)
            print "cost:"
            print cost
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

#        #For each weight individually, approximate gradient.
#        for i in range(len(theta)):
#            #Create modified theta vectors using e = [0,0,0,0,i=1,0,0].                
#            e = np.zeros(np.shape(theta))
#            e[i] = epsilon            
#            thetaP = (theta + e)
#            thetaN = (theta - e)
#
#            costP = 0.0
#            costN = 0.0
#            
#            #calculate sum of cost for entire dataset.
#
#            wdP = wd_lambda/2.0*np.sum(np.power(thetaP,2.0))            
#            self.deFlattenWeights(thetaP)             
#            for j in range(len(x)):
#                self.feedForward(x[j])
#                #self.feedBack(y[j])
#                costP += 0.5*np.power(np.linalg.norm(self.layers[-1].a-y[j], ord=2),2.0)
#
#            wdN = wd_lambda/2.0*np.sum(np.power(thetaN,2.0))                
#            self.deFlattenWeights(thetaN)                
#            for j in range(len(x)):
#                self.feedForward(x[j])
#                #self.feedBack(y[i])                
#                costN += 0.5*np.power(np.linalg.norm(self.layers[-1].a-y[j], ord=2),2.0)
#
#        
#
#            #calculate approximate gradient from consts
#            grad[i] = ( (costP/double(len(x)) + wdP) - (costN/double(len(x)) + wdN) )/(2.0*epsilon)             
            
        #####################################################
            
#        #Update weight matrices.
#        for i in range(len(self.connections)):
#            self.connections[i].W = self.connections[i].W - lr_alpha*(self.connections[i].delta_W/double(len(x)) + wd_lambda*self.connections[i].W)
#            self.connections[i].W_b = self.connections[i].W_b - lr_alpha*self.connections[i].delta_W_b/double(len(x))
#   
        
        weightSum =  np.sum(np.power(theta,2.0))
        
        finalCost = cost/double(len(x)) + wd_lambda/2.0*weightSum

        return (finalCost,np.abs(nabla-grad))
        
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
        
        init_epsilon = 0.9#0.01 #going from 0.01 to 1.01 made it possible to train an xor function. Before, with all values at approx 0.00001, all outputs just went towards 0.5, i.e. avrg output.
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