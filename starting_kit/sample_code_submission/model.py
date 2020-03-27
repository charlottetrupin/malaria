'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''

import pickle # Pour enregistrer et charger modèle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile # Fonction fichier
from sklearn.base import BaseEstimator # Interface d'un estimator
from sklearn.ensemble import RandomForestClassifier # Modèle choisi
from sklearn.decomposition import PCA # Preprocessing
from sklearn.metrics import roc_auc_score # Méthode de calcul du score
from sklearn.metrics import make_scorer # Conversion de la fonction de score
from sklearn.model_selection import RandomizedSearchCV # Optimisation hyperparamètres
import matplotlib.pyplot as plt; import seaborn as sns; sns.set() # Affichage graphique
from sklearn.metrics import confusion_matrix # Matrice de confusion


class model (BaseEstimator):
    def __init__(self, classifier=RandomForestClassifier(), n_pca = 4):
   
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
         Args :
            classifier : classifier we will use for making our predictions
            n_pca : argument for preprocessing
        '''
        self.num_train_samples=11077 # Nombre de données entraînement
        self.num_feat=6 # Nombre de features
        self.num_labels=1 # Nombre de labels
        self.is_trained=False # Etat de l'apprentissage
        self.pca = PCA(n_components = n_pca) # Preprocessing
        self.classifier = classifier # Modèle de classification
        self.scoring_function = roc_auc_score # Méthode de calcul du score
        
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
            print("Error: number of samples in X and y do not match!")
        
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
            print("Error: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        
        return self.classifier.predict(X)
    
    def optimize(self, X, y, n_iter=100):
        """ 
        Optimise le classifieur en cherchant les meilleurs hyperparamètres
        Args:
            X : jeu de données d'entraînement
            y : labels correspondants
            n_iter : nombre de combinaisons testées (default=100)
        """
        # Paramètres à tester
        parameters_tree={'classifier__criterion':["gini", "entropy"], 
                         'classifier__n_estimators':[i for i in range(10,300,10)], 
                         'classifier__max_depth':[i for i in range(1,10)]+[None], 
                         'classifier__min_samples_split':[i for i in range(2,5)], 
                         'classifier__min_samples_leaf':[i for i in range(1,5)],
                         'classifier__random_state':[i for i in range(1,100)]}
        # Grille de recherche en utilisant toute la puissance processeur et paramétrée avec la fonction de score
        grid = RandomizedSearchCV(F, parameters_tree, scoring=make_scorer(scoring_function), n_jobs=-1, n_iter=n_iter)
        # Lancement des entrainements
        grid.fit(X, y)
        # Meilleurs hyperparamètres
        self.classifier = grid.best_estimator_
        
    def score(self, X, y):
        """ 
        Calcul du score du modèle actuel 
        sur un jeu de données quelconque
        Args:
            X : jeu de données
            y : labels correspondants
        """
        # Score du modèle
        return scoring_function(y, self.classifier.predict(X))
    
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        """ Rechargement du modèle préalablement enregistré """
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
                print("Model reloaded from: " + modelfile)
        return self

    def transform(self, X):
        """ Preprocessing du jeu de données X """
        # FONCTION A ADAPTER
        # Il faudra changer le nombre de features également je suppose
        liste = []
        estimator = RandomForestClassifier(n_estimators = 10) # <-- pourquoi créer un autre classifieur ???
        estimator.fit(X_train)
        for i in range(data.shape[0]): # <-- erreur
            if estimator.predict(X)[i] != -1 :
                liste.append(i)
        X = X[liste,:]
        X_1 = self.mod.transform(X) # <-- self.classifier.transform(X) ?
        X_2 = self.pca.transform(X)
        return X_new # <-- erreur
    
    def confusion_matrix(self, X, y):
        """ 
        Affiche la matrice de confusion de préférence
        sur l'ensemble de test
        Args:
            X : jeu de données
            y : labels correspondants
        """
        ax = plt.subplot()
        # annot=True to annotate cells
        sns.heatmap(confusion_matrix(y, self.classifier.predict(X)), annot=True, fmt='g', ax=ax)

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Parasitized', 'Uninfected']);
        ax.yaxis.set_ticklabels(['Parasitized', 'Uninfected']);
        plt.show()

