# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.layers.normalization import BatchNormalization
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import itertools
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


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
batch_size = 10
n_epochs = 80

#class_weight = {0: 1., 1: 5.}
#
# load training dataset
cv_dataset = np.loadtxt(trn_file, delimiter=",")
# split into input (X) and output (Y) variables
X = cv_dataset[:,1:401].reshape(len(cv_dataset),1,20,20)
Y = cv_dataset[:,0]

# load testing dataset
ind_dataset = np.loadtxt(tst_file, delimiter=",")
# split into input (X) and output (Y) variables
X1 = ind_dataset[:,1:401].reshape(len(ind_dataset),1,20,20)
Y1 = ind_dataset[:,0]
true_labels_ind = np.asarray(Y1)

Y1 = np_utils.to_categorical(Y1,nb_classes)
#print(Y1)

def cnn_model():
    model = Sequential()

    model.add(ZeroPadding2D((1,1), input_shape = (1,20,20)))
    model.add(Conv2D(32, (nb_kernels, nb_kernels), activation='relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), data_format="channels_first"))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (nb_kernels, nb_kernels), activation='relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), data_format="channels_first"))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (nb_kernels, nb_kernels), activation='relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), data_format="channels_first"))

    ## add the model on top of the convolutional base
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
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
    model.fit(X[train], np_utils.to_categorical(Y[train],nb_classes), epochs=n_epochs, batch_size=batch_size, verbose=0)
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
print(model.summary())
##plot_model(model, to_file='model.png')
#
#filepath = "weights.best.hdf5"
#checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True)
## balance data
model.fit(X, np_utils.to_categorical(Y, nb_classes), epochs=n_epochs, batch_size=batch_size, verbose=0, class_weight='auto')
### evaluate the model
scores = model.evaluate(X1, Y1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#
#model.load_weights(filepath)
predictions = model.predict_classes(X1)

cnf_matrix = confusion_matrix(true_labels_ind, predictions)

#print(cnf_matrix)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['non-clathrin','clathrin'])
