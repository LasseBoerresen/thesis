#"""
#Compute and display a spectrogram.
#Give WAV file as input
#"""
import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import sys

import os
import math
import time
import scipy as sp
from scipy import ndimage, misc
import random
import skimage
from skimage import data, filter, io

import pickle



class soundCleaver:
    def __init__(self):
        self.sr = 8000#8000        
        #self.numImages = 1
        self.numPatches = 200 #5000
        print('num samples: \n',int(2.0*self.sr))#int(1*self.sr))
        self.sizePatches = int(2.0*self.sr)#in seconds. Multiply with sampleRate, sr, to get number of saples in patch    #8 must be equal number, to be able to split in half later
        self.soundDataBase = []
        self.soundLabelDataBase = []


#        self.addToSoundDataBase('Cough test.wav',0)        
#        self.addToSoundDataBase('background2.wav',0)        
#        self.addToSoundDataBase('Background3.wav',0)        
#        self.addToSoundDataBase('Background4.wav',0)        
#        self.addToSoundDataBase('Background video.wav',0)


        self.addToSoundDataBase('Bachround5.wav',0)   
#        self.addToSoundDataBase('Passive.wav',1)        
        self.addToSoundDataBase('Passiv2.wav',1)        
        self.addToSoundDataBase('Passive reading 0.wav',1)        
#        self.addToSoundDataBase('happy.wav',2)        
        self.addToSoundDataBase('Happy 1.wav',2)        
        self.addToSoundDataBase('Happy reading 1.wav',2)        
    
 
#       self.addToSoundDataBase('angry.wav',3)
        self.addToSoundDataBase('Angry 1.wav',3)
        self.addToSoundDataBase('Angry reading2.wav',3)
        self.addToSoundDataBase('Angry reading3.wav',3)


                
        


        

#        self.srDataBase = []
        self.patchDataBase = []
        self.patchLabelDataBase = []
        for i in range(self.numPatches):
             #should be random num
            j = random.randint(0,len(self.soundDataBase)-1)
            
            randPosX = random.randint(self.sizePatches/2, len(self.soundDataBase[j])-self.sizePatches/2)            
            #randPosY = random.randint(self.sizePatches/2, self.imageDataBase[self.numImages-1].shape[1]-self.sizePatches/2)                        
            self.patchDataBase.append(self.soundDataBase[j][randPosX-self.sizePatches/2 : randPosX+self.sizePatches/2])
            self.patchLabelDataBase.append(self.soundLabelDataBase[j])
            
        self.concSoundArray = np.zeros((self.sizePatches,self.numPatches))
        self.concatenateSoundPatches()
        print(self.patchDataBase[5])
        
        print('DONE CREATING DATASET')
#        print(patchDataBase)
#        plt.imshow(patchDataBase[0])    
        

    def addToSoundDataBase(self, soundFile, classLabel):
        wavfile = soundFile #sys.argv[1]

        sr,x = scipy.io.wavfile.read(wavfile)
#        print("sampling rate = {} Hz, length = {} samples, channels = {}".format(sr, *x.shape))
        
#        print sr
        print('x.shape',x.shape)
        print('len(x.shape)',len(x.shape))
        if len(x.shape) == 2:
            xLeft = []
            xRight = []
            for i in range(len(x)):
                xLeft.append(x[i][0])
                xRight.append(x[i][1])
            
            self.soundDataBase.append(xLeft)
        else:
            self.soundDataBase.append(x)

        self.soundLabelDataBase.append(classLabel)
#        self.srDataBase.append(sr)
        

    #to use the images most easily for inputs to a NN, they are unravelled from 8x8 to 64x1,
    #and then concatenated so each column is another image.        
    def concatenateSoundPatches(self):        
#        imgArray = np.array([[]])
        concSoundArrayT = np.transpose(self.concSoundArray)
        
        for i in range(self.numPatches):     
            concSoundArrayT[i] = self.patchDataBase[i]
        self.concSoundArray = np.transpose(concSoundArrayT)


    def spectrogramFromPatch(self,patch):
        nstep = int(self.sr * 0.01)
        nwin  = int(self.sr * 0.03)
        nfft = nwin
        
        
        window = np.hamming(nwin)
        
        ## will take windows x[n1:n2].  generate
        ## and loop over n2 such that all frames
        ## fit within the waveform
        nn = range(nwin, len(patch), nstep)
        
        
        
        X = np.zeros( (len(nn), nfft/2  ) )
#        print('nstep',nstep)        
#        print('nwin',nwin)        
##        print('nn',nn)        
#        print('X.shape',X.shape)        
        
        for i,n in enumerate(nn):
            xseg = patch[n-nwin:n]
            z = np.fft.fft(window * xseg, nfft)
            X[i,:] = np.log(np.abs(z[:nfft/2]))

        return X.T


def plotSoundArray(xLeft,sr,fromT, toT):
        
    y = xLeft[int(sr*fromT):int(sr*fromT)+int(sr*toT)]
    x = np.array(range(len(y)))#/sr
    x = x/float(sr)
    print('x: ',  x)
    print('y: ',  y)
    
    figure(1)
    plt.plot(x,y)

#
#def main():
#    sc = soundCleaver()
#
#    
#    patch = sc.patchDataBase[0]
#
##    plotSoundArray(patch,44100,0.0,0.1)    
#    
##    lena = sp.misc.imread('lena.png')
##    print 'lena',lena
#    patchSpec = sc.spectrogramFromPatch(patch)
##
##    print 'patchSpec.shape', patchSpec.shape
###    NDAPatchSpec = np.array(patchSpec)
##    
##    patchSpecInt = np.zeros(patchSpec.shape, dtype=int)
##    for i in range(patchSpec.shape[0]):
##        for j in range(patchSpec.shape[1]):
##            patchSpecInt[i][j] = int(patchSpec[i][j])
###    print NDAPatchSpec
##    print 'patchSpecInt', patchSpecInt
##    smallPatchSpec = sp.misc.imresize(patchSpecInt, 10)
#    
#    plt.imshow(patchSpec, interpolation='nearest', origin='lower')
#    plt.show()
#    
#    
#    
#    
#if __name__ == '__main__':
#    main()