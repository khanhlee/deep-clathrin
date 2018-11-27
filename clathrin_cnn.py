# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:49:56 2018

@author: khanhle
"""



# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print(__doc__)

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from scipy import interp

#define params
#trn_file = sys.argv[1]
#tst_file = sys.argv[2]
trn_file = 'dataset/pssm.cv.csv'
tst_file = 'dataset/pssm.ind.csv'

num_features = 400
nb_classes = 2
nb_kernels = 3
nb_pools = 2
window_sizes = 20

#class_weight = {0: 1., 1: 5.}
#
# load training dataset
cv_dataset = np.loadtxt(trn_file, delimiter=",")
# split into input (X) and output (Y) variables
X = cv_dataset[:,0:window_sizes*20].reshape(len(cv_dataset),1,20,window_sizes)
Y = cv_dataset[:,window_sizes*20]

#Y = np_utils.to_categorical(Y,nb_classes)
#print X,Y
#nb_classes = Y.shape[1]
#print nb_classes
#
# load testing dataset
ind_dataset = np.loadtxt(tst_file, delimiter=",")
# split into input (X) and output (Y) variables
X1 = ind_dataset[:,0:window_sizes*20].reshape(len(ind_dataset),1,20,window_sizes)
Y1 = ind_dataset[:,window_sizes*20]
true_labels_ind = np.asarray(Y1)

Y1 = np_utils.to_categorical(Y1,nb_classes)
#print(Y1)



#i = 4600
#plt.imshow(X[i,0], interpolation='nearest')
#print('label : ', Y[i,:])

def cnn_model():
    model = Sequential()

#    model.add(Dropout(0.2, input_shape = (1,20,window_sizes)))
    model.add(ZeroPadding2D((1,1), input_shape = (1,20,window_sizes)))
    model.add(Conv2D(32, (nb_kernels, nb_kernels)))
    model.add(Activation('relu'))
#    model.add(Dropout(0.2))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), data_format="channels_first"))
    
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(64, (nb_kernels, nb_kernels)))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), data_format="channels_first"))
#    
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(128, (nb_kernels, nb_kernels)))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), data_format="channels_first"))
#    
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(256, (nb_kernels, nb_kernels)))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), data_format="channels_first"))

    ## add the model on top of the convolutional base
    #model.add(top_model)
    model.add(Flatten())
#    model.add(Dropout(0.1))
    model.add(Dense(32))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
#    model.add(Dropout(0.2))

    model.add(Dense(nb_classes))
    #model.add(BatchNormalization())
    model.add(Activation('softmax'))

    # f = open('model_summary.txt','w')
    # f.write(str(model.summary()))
    # f.close()

    #model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

#
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores = []
# SVM
import timeit

start = timeit.default_timer()

# CNN
for train, test in kfold.split(X, Y):
    model = cnn_model()   
    ## evaluate the model
    model.fit(X[train], np_utils.to_categorical(Y[train],nb_classes), epochs=50, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], np_utils.to_categorical(Y[test],nb_classes), verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    #prediction
    #model.load_weights(filepath)
    true_labels = np.asarray(Y[test])
    predictions = model.predict_classes(X[test])
    print(confusion_matrix(true_labels, predictions))

stop = timeit.default_timer()

print('Time: ', stop - start)
    

#plot_filters(model.layers[0],32,1)
# Fit the model
# save best weights
model = cnn_model()
##plot_model(model, to_file='model.png')
#
#filepath = "weights.best.hdf5"
#checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True)
earstop = EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None)
## balance data
model.fit(X, np_utils.to_categorical(Y, nb_classes), epochs=50, batch_size=10, verbose=0, class_weight='auto')
### evaluate the model
scores = model.evaluate(X1, Y1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#
#model.load_weights(filepath)
predictions = model.predict_classes(X1)
print(confusion_matrix(true_labels_ind, predictions))
