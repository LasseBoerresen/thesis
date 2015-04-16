#Task: separate 2 sinus signals.
#
#
#
#filter signals, and 
#
#Separating 
#
#
#Separating complex signals,  
#    could be done by training deep features, 
#    and filtering for theese features individually, 
#    to get direction information, 
#    and then use directional information to group together features, 
#    and with that cluster... 
#
#
#
#Filtering frequencies is relatively easy, (Calculate inverse fft and apply to time domain data)
#    but what about filtering for a certain pattern from a spectrogram? (apply individual ifft's for each window sequentially)
#    Should direction be obtained in time(amplitude) or frequency domain. Well, spectrogram is a freq/time domain.
#
#
#To analyse a set of features for possible 
#
#Clustering works on multiple datapoints. 



#1 Get filtered versions of input
#2 find direction for filtered input
#3 match directions to find what components are combined.

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
        self.sr = 48000#8000        
        #self.numImages = 1
        self.numPatches = 200 #5000
#        print('num samples: \n',int(2.0*self.sr))#int(1*self.sr))
        self.sizePatches = int(1.00*self.sr)#in seconds. Multiply with sampleRate, sr, to get number of saples in patch    #8 must be equal number, to be able to split in half later
        self.soundDataBase = []
        self.soundLabelDataBase = []


#        self.addToSoundDataBase('Cough test.wav',0)        
#        self.addToSoundDataBase('background2.wav',0)        
#        self.addToSoundDataBase('Background3.wav',0)        
#        self.addToSoundDataBase('Background4.wav',0)        
#        self.addToSoundDataBase('Background video.wav',0)


#        self.addToSoundDataBase('Bachround5.wav',0)   
##        self.addToSoundDataBase('Passive.wav',1)        
#        self.addToSoundDataBase('Passiv2.wav',1)        
#        self.addToSoundDataBase('Passive reading 0.wav',1)        
##        self.addToSoundDataBase('happy.wav',2)        
#        self.addToSoundDataBase('Happy 1.wav',2)        
#        self.addToSoundDataBase('Happy reading 1.wav',2)        
#    
# 
##       self.addToSoundDataBase('angry.wav',3)
#        self.addToSoundDataBase('Angry 1.wav',3)
#        self.addToSoundDataBase('Angry reading2.wav',3)
#        self.addToSoundDataBase('Angry reading3.wav',3)

        self.addToSoundDataBase('sineMix.wav',0)
                
        


        

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
        #print(self.patchDataBase[5])
        
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

    #Given an angle as input, phase shift a mono signal to create a simulated directional stereo sound signal.
    def phaseShiftMono(self,soundArr,theta):

        speedOfSound = 344.0
        a = 0.013 #Without lizard ears, could be much bigger.
        b = 1.000        
        
        # calculation of distance to each microphone, given 'b' distance from center.
        distL = np.sqrt((a/2.0)**2.0 + b**2.0 - 2*(a/2.0)*b*np.cos(np.pi/2.0 - theta))
        distR = np.sqrt((a/2.0)**2.0 + b**2.0 - 2*(a/2.0)*b*np.cos(np.pi/2.0 + theta))

        #calculate time difference
        #if direction is left, timeDiff is positive
        distDiff = distR - distL
        timeDiff = np.abs(distDiff/speedOfSound)
        
        #calculate difference in number of samples, and create small silent sound piece, to add.
        sampleShiftSize = int(self.sr*timeDiff)
        sampleShift = np.zeros(sampleShiftSize) 

        #initiate the stereo channels of correct length
        right = np.ndarray(sampleShiftSize+len(soundArr))
        left = np.ndarray(sampleShiftSize+len(soundArr))        
        
        #dependent on positive or negative angle, add the sampleShift ad appropriate end of sample.
        if theta > 0.0:
            right = np.concatenate((sampleShift,soundArr))
            left = np.concatenate((soundArr,sampleShift))
        else:
            left = np.concatenate((sampleShift,soundArr))
            right = np.concatenate((soundArr,sampleShift))
        
        return (right,left)                
        
        
        
    # Find characteristic peaks, and calculate distances, or feed both channels into nn at train it to find direction.     
    #Brute force. substract snippet from longer sample to find minimum difference, this gives time difference.
    def directionFromTimeDiff(self,tDiff):
        speedOfSound = 344.0
        a = 0.013 #Without lizard ears, could be much bigger.
        b = 1.000 
        
        
        dDiff = tDiff*speedOfSound
        #Distance is alwasy 
        distR = dDiff/2.0
        distL = dDiff/2.0*(-1.0)
        
        theta = np.pi/2.0 - np.arccos(((a/2.0)**2.0 + b**2.0 - distL)/2.0*(a/2.0)*b)        
        
        return theta
        
  
    # Could use RanSaC to find first position, then that 3D alignment code. 
    #   However, I only need to check along one axis, time, frequency or amplitude is not shiftet.... amplitude could however be shifted when including this aspect. However, it is not so much shifted as it is scaled. amplitude will not go below zero... 
    # Find overlap in frequency domain, i.e. spectrogram. 
    #OBS: if continous sound with not change in frequency over time.  
    def timeDiffFromStereo(self, rightImg, leftImg):
        winSize = 10
        sgWinSize = 30 #ms

        #DESTROY 
#        rightImg = np.ndarray([100,500])
#        leftImg = np.ndarray([100,500])        
        
        #extract window of lenth winSize along time axis, from center, to maximize chance of finding overlap.
        window = rightImg[:,len(rightImg[0])/2-winSize/2:len(rightImg[0])/2+winSize/2]

        minVal = double('inf')
        minPos
        #find minimum squeared difference between window and spectrogram
        for i in range(len(rightImg[0])-winSize):
            #difference is found as vector norm of all differences.
            diff = 0.5*np.power(np.linalg.norm(np.reshape(leftImg[:,i:i+winSize] - window, [len(window[0])*len(window[1])]), ord=2), 2.0)
            if diff < minVal:
                minVal = diff
                minPos = i
        
        #if direction is from left, tDiff is positive
        tDiff = (len(rightImg[0])/2 - minPos)*sgWinSize
        return tDiff
        
        
    # the above function can only find direction of sound where frequencies change. So at start of sine wave, or of speach...
    # Direction is much better estimated using difference in sound amplitude due to microphone directionality and sound blocking.
    #   To get function of sound amplitude difference at varieing directions and frequencies, simply measure with real setup.
    #Firstly, simply use difference in RMS    
    def 
    
        
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

        #creates list of windiw indecies, specifically, the sample at which they end.
        nn = range(nwin, len(patch), nstep)
        
        
        #Initialize spectrum matrix, with len(nn) = number of window, and nfft/2 = how many frequency bins.
        X = np.zeros( (len(nn), nfft/2  ) )
       
        
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

def fftFromPatch(patch):
        window = np.hamming(len(patch))
        z = np.fft.fft(window * patch)
        print "type(z):"
        print type(z)
        print 
#        np.fft.ifft()
        
        return (np.log(np.abs(z[:len(patch)/2])), np.imag(z))

def plotFFT(fft, fig=1):
    figure(fig)
    x = range(len(fft))
    for i in range(len(fft)):
        x[i] = x[i]
    plt.plot(x,fft)


def gaussian(mean,sigma,length):
    x = range(0,length)
    for i in range(len(x)):
        hej = math.exp(-(x[i]-mean)*(x[i]-mean)/(2.0*sigma*sigma))
        x[i] = (1.0/(sigma*math.sqrt(2.0*math.pi)))*hej
    return x
 
def normalizeList(inList):
    ma = max(inList)
    mi = min(inList)
    diff = ma-mi
    outList = range(len(inList))
    for i in range(len(inList)):
        outList[i] = (inList[i] - mi)/diff
    return outList

def negativeNormGaussian(mean,sigma,length):
    g = gaussian(mean,sigma,length)
    g = normalizeList(g)
    gNeg = g    
    for i in range(len(g)):
        gNeg[i] = 1-g[i]
    return gNeg


def findFFTPeaks(z):
    firstMax = max(z)
    z_new = []
    for i in range(len(z)):
        z_new.append(z[i])
        
    result = []
    while 1:
        m =  max(z_new)
        i_m = [i for i, j in enumerate(z_new) if j == m]
        g = negativeNormGaussian(i_m[0],40,len(z_new))
        for i in range(len(z_new)):
            z_new[i] = z_new[i]*g[i]
        
        result.append((i_m[0],m))        
        if max(z_new) < firstMax/2.0:
            break
    return (z_new,result)
        
def main():
    sc = soundCleaver()

    
    patch = sc.patchDataBase[0]

    #window = np.hamming(len(patch))

    zz = np.fft.rfft(patch)#*window)

    

    zz_log = np.log(np.abs(zz))
    zz_peaks_removed, result = findFFTPeaks(zz_log)

    print result
    
    zz_fil_list = []
    zz_filtered = np.ndarray(np.shape(zz))
    #for i in range(len(zz)):
        
    g = gaussian(result[0][0],len(zz)/32,len(zz))
    g = normalizeList(g)

    plotFFT(g,1212)

    


    #multiply data with filter elementwise
    for j in range(len(zz)):
        zz_filtered[j] = zz[j]*g[j]

    plotFFT(zz_filtered,1)

    xx = np.fft.irfft(zz_filtered)

    xxx = np.ndarray(np.shape(patch),dtype=int32)       
    
    for i in range(len(xx)):
        xxx[i] = int(xx[i])

    sp.io.wavfile.write("patch.wav", 48000, patch)
    
    sp.io.wavfile.write("fft.wav", 48000, xxx)


#    #transform patch into frequency domain
#    z,zImg = fftFromPatch(patch)
#
#    #get peak positions
#    z_peaks_removed, result = findFFTPeaks(z)
#    
#    print
#    print "result:"
#    print result
#    print
#    
#    
#    #Get list of filtered z for each peak 
#    z_filtered = [] 
#    z_filtered_list = []
#    for i in range(len(result)):
#        #create gaussian filter in appropriate position
#        g = gaussian(result[i][0],40,len(z))
#        g = normalizeList(g)
#        #multiply data with filter elementwise
#        for j in range(len(z)):
#            z_filtered.append(z[j]*g[j])
#        #add filtered data to list    
#        z_filtered_list.append(np.array(z_filtered))
#        #reset for next peak        
#        z_filtered = []        
#
#    z_filtered_list[0] = z
#    plotFFT(z_filtered_list[0], 1212)
#    
#
#    #invert the initial log() transform 
#    z_fil_exp = np.exp(z_filtered_list[0])
#
#    plotFFT(z_fil_exp, 1213)
#
#
#    # mirror real part of frequency domain to get negative frequencies needed for inverse fft
#    z_fil_mir = []    
#    for i in range(len(z_fil_exp) ):
#        z_fil_mir.append(z_fil_exp[i])    
#    for i in range(len(z_fil_exp)-1,0,-1 ):
#        z_fil_mir.append(z_fil_exp[i])    
#
#
#
#    plotFFT(z_fil_mir,23)
#    # convert to np array from list
#    z_fil_mir = np.array(z_fil_mir)
#    
#
#
#    z_fil_comp = []
#    for i in range(len(z_fil_mir)):
#        z_fil_comp.append(np.complex(z_fil_mir[i],zImg[i]))
#    x_fil = np.fft.ifft(z_fil_comp)
#
#    print "type(x_fil[0])"
#    print type(np.real(x_fil)[0])
#
#    x_fil_real = np.real(x_fil)
#
#    x_fil_real_int = np.ndarray(np.shape(x_fil_real), dtype=int32)
#    for i in range(len(x_fil_real)):
#        x_fil_real_int[i] = int(x_fil_real[i]) 
#    
#    print
#    print
#    print "type(x_fil_real_int[0])"
#    print type(x_fil_real_int[0])
#    
#    plotFFT(x_fil_real_int, 122)
#    
#    sp.io.wavfile.write("filtered.wav", 48000, x_fil_real_int)
    
    
#    plotSoundArray(patch,48000,0.0,0.01)    
    
#    lena = sp.misc.imread('lena.png')
#    print 'lena',lena

    #patchSpec = sc.spectrogramFromPatch(patch)

#
#    print 'patchSpec.shape', patchSpec.shape
##    NDAPatchSpec = np.array(patchSpec)
#    
#    patchSpecInt = np.zeros(patchSpec.shape, dtype=int)
#    for i in range(patchSpec.shape[0]):
#        for j in range(patchSpec.shape[1]):
#            patchSpecInt[i][j] = int(patchSpec[i][j])
##    print NDAPatchSpec
#    print 'patchSpecInt', patchSpecInt
#    smallPatchSpec = sp.misc.imresize(patchSpecInt, 10)
    
    #plt.imshow(patchSpec, interpolation='nearest', origin='lower')
    plt.show()
    
    
    
    
if __name__ == '__main__':
    main()