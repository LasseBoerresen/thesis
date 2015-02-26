from __future__ import print_function

print(__doc__)

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

import waveSpec

import skimage
import skimage.transform

from sklearn import decomposition

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


###############################################################################
# Setting up

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def main():
    sc = waveSpec.soundCleaver()

    
#    patch = sc.patchDataBase[0]
    
#    lena = sp.misc.imread('lena.png')
#    print 'lena',lena
#    patchSpec = sc.spectrogramFromPatch(patch)

    
#    plt.imshow(patchSpec, interpolation='nearest', origin='lower')
#    plt.show()



    # Load Data
#    digits = datasets.load_digits()
#    print(np.asarray(digits.data, 'float32').shape)
#    print('digits as array', np.asarray(digits.data, 'float32'))
#    print('digits.data', digits.data)
#    
#    print('sc.patchDataBase.shape', np.array(sc.patchDataBase).shape)
    

    Xp = np.asarray(sc.patchDataBase, 'float32')#np.asarray(digits.data, 'float32')
    Yp = np.asarray(sc.patchLabelDataBase)#nudge_dataset(X, digits.target)
    
    print('Xp.shape:\n', np.array(Xp).shape)
    
    print('STARTING CONVERTING TO SPECTROGRAMS')
    XpSpec = []
    
    for i in range(len(Xp)):
        
        spec = sc.spectrogramFromPatch(Xp[i])
        h,w = spec.shape
#        print('Xp[i]',Xp[i])
#        print('spec',spec)
#        plt.imshow(spec)
#        print('spec.shape: \n', spec.shape)    
#        figure(1)        
#        plt.imshow(spec)
#        smallSpec = skimage.transform.rescale(spec,0.1)
#        print('smallSpec.shape: \n', smallSpec.shape)
#        figure(2)
#        plt.imshow(smallSpec)

        XpSpec.append(spec.ravel() )
    print('DONE CONVERTING TO SPECTROGRAMS')
    
#    Xp = skimage.transform.rescale( XpSpec,10)
    


    X_train, X_test, Y_train, Y_test = train_test_split(XpSpec, Yp, test_size=0.2, random_state=0)

#    pca = decomposition.RandomizedPCA(n_components=256, whiten = True)
#    pca.fit(X_train)
#    
#    X_train = pca.transform(X_train)
#    X_test = pca.transform(X_test)
    
    
    #Xp = (Xp - np.min(Xp, 0)) / (np.max(Xp, 0) + 0.0001)  # 0-1 scaling

    
  
    

#    # Load Data
#    digits = datasets.load_digits()
#    X = np.asarray(digits.data, 'float32')
#    X, Y = nudge_dataset(X, digits.target)
#    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
#
#    Y=Yp
#    X=Xp
#    print('X\n', X)
#    
#    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#    
    # Models we will use
#    param_grid = {'svm__C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#    clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)        

    n_components = 128


    print("Extracting the top %d spectrograms from %d spectrograms"
      % (n_components, X_train.shape[0]))


    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    
    print("pca.explained_variance_ratio",pca.explained_variance_ratio_)
    
    eigenfaces = pca.components_.reshape((n_components, h, w))
    print("testetstwetet", len(pca.components_[0]))    
    s = ""    
    
    f = open("pca_components200.txt",'w')
    for i in range(len(pca.components_)):
        for j in range(len(pca.components_[i])):
              s = s + str(pca.components_[i][j]) + ' '
        f.write(s)
        s = ""
    f.close()
    
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))    

    
    svm2 = SVC(kernel='rbf', class_weight='auto', C = 1, gamma = 0.01)
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)
#    pca = decomposition.RandomizedPCA(n_components=64, whiten = True)

    svm = Pipeline(steps=[('rbm',rbm), ('svm2',svm2)])

    param_grid = {'pca__n_components': [32 ,64, 128],'svm__C': [1e3, 1e4, 1e5], 'svm__gamma': [0.0001, 0.001 , 0.1] }
#    clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)        

#    classifier = GridSearchCV(Pipeline(steps=[('pca',pca), ('svm',svm)]), param_grid=param_grid, verbose=10)

    
#    classifier = Pipeline(steps=[('pca',pca), ('svm',svm)])

    
#    logisticClassifier = Pipeline(steps=[('pca',pca), ('logistic',logistic)])
#    classifier = Pipeline(steps=[('pca',pca),('rbm', rbm), ('logistic', logistic)])
    
    ###############################################################################
    # Training
    
    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 100
    logistic.C = 6000.0
    
    # Training RBM-Logistic Pipeline
#    classifier.fit(X_train, Y_train)
#    logisticClassifier.fit(X_train, Y_train)
    # Training Logistic regression
    param_grid_svm = {'C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5], 'gamma': [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05 , 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0] }  
    param_grid_svm = {'svm2__C': [1e-1, 1e0, 1e1], 'svm2__gamma': [0.005, 0.01, 0.05] }  

    param_grid_log = {'C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]}
    param_grid_log = {'C': [1e-1, 1e0, 1e1]}

    svm_classifier = GridSearchCV(svm, param_grid_svm, verbose=10, n_jobs = 4)
    svm_classifier.fit(X_train, Y_train)
    
    print("Best parameters set found on development set:")
    print()
    print(svm_classifier.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in svm_classifier.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))

    print()    
    print()    
    print()    
    
    logistic_classifier = GridSearchCV(logistic, param_grid_log,verbose=10, n_jobs = 2)
    logistic_classifier.fit(X_train, Y_train)
    print("Best parameters set found on development set:")
    print()
    print(logistic_classifier.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in logistic_classifier.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    
    ###############################################################################
    # Evaluation
    print()
    print("svm train using pca features:\n%s\n" % (
        metrics.classification_report(            
            Y_train,
            svm_classifier.predict(X_train))))
             
    print()
    print("svm test using pca features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            svm_classifier.predict(X_test))))
    
     
#    print()
#    print("Logistic regression using PCA features:\n%s\n" % (
#        metrics.classification_report(
#            Y_test,
#            classifier.predict(X_test))))    
    
    print("Logistic regression using pca pixel features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            logistic_classifier.predict(X_test))))
    
    ###############################################################################
    #Plotting
    
#    plt.figure(figsize=(4.2, 4))
#    for i, comp in enumerate(rbm.components_):
#        plt.subplot(10, 10, i + 1)
#        plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
#                   interpolation='nearest')
#        plt.xticks(())
#        plt.yticks(())
#    plt.suptitle('100 components extracted by RBM', fontsize=16)
#    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
#    
#    plt.show()


    
    
    eigenface_titles = ["spectrogram %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w, 6,6)

plt.show()
    
if __name__ == '__main__':
    main()