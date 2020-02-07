'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''

import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

class model (BaseEstimator):
   def __init__(self, classifier=RandomForestClassifier(random_state=42)):
   
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
         args :
            classifier : classifier we will use for making our predictions
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        self.classifier = classifier
        
        
   def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        
        '''
        
        
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        
        
        
        self.classifier.fit(X, np.ravel(y))
        self.is_trained=True
        
        return self
   
   def predict(self, X):
        '''
        This function  provides predictions of labels on (test) data
        '''
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        
        
        return self.classifier.predict(X)
   def save(self, path="./"):
      pickle.dump(self, open(path + '_model.pickle', "wb"))

   def load(self, path="./"):
      modelfile = path + '_model.pickle'
      if isfile(modelfile):
         with open(modelfile, 'rb') as f:
            self = pickle.load(f)
         print("Model reloaded from: " + modelfile)
      return self

