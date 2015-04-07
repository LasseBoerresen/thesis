import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

#from theano import *
#import theano.tensor as T
#from theano import function


a = 0.013
b = 1.00

theta = np.pi*3.0/8.0


res = np.arange(-np.pi/2.0, np.pi/2.0+np.pi/200.0, np.pi/(2.0*100))


distDiffList = []
for i in range(len(res)):
    theta = res[i]
    distL = np.sqrt((a/2.0)**2.0 + b**2.0 - 2*(a/2.0)*b*np.cos(np.pi/2.0 - theta))
    distR = np.sqrt((a/2.0)**2.0 + b**2.0 - 2*(a/2.0)*b*np.cos(np.pi/2.0 + theta))
    
    distDiffList.append(distR - distL)


plt.plot(res, distDiffList)

#plt.plot(sin(res)*0.05)
