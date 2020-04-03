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
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA # Preprocessing
from sklearn.metrics import roc_auc_score # Méthode de calcul du score
from sklearn.metrics import make_scorer # Conversion de la fonction de score
from sklearn.model_selection import RandomizedSearchCV # Optimisation hyperparamètres
import matplotlib.pyplot as plt; import seaborn as sns; sns.set() # Affichage graphique
from sklearn.metrics import confusion_matrix # Matrice de confusion

class preprocess:
    def __init__(self, n_pca=4):
        ''' Classe pour le preprocessing
        '''
        self.is_trained = False # Etat de l'apprentissage
        self.n_pca = n_pca
        self.pca = PCA(n_components=n_pca) # Preprocessing
        # pour le moment on enleve l'isolation forest, il faut trouver un parametre qui reduit le temps d'exécution
        #self.estimator = IsolationForest(n_estimators=10) # outlier detection
        
    def fit(self, X):
        self.pca.fit(X)
        #self.estimator.fit(X)
        self.is_trained = True
        
    def transform(self, X):
        """ Preprocessing du jeu de données X """
        # Il faudra changer le nombre de features également je suppose
        #liste = []
        #for i in range(X.shape[0]):
        #    if self.estimator.predict(X)[i] != -1 :
        #        liste.append(i)
        #X = X[liste, :]
        #X = self.estimator.transform(X) # delete outliers
        X = self.pca.transform(X) # reduce dimension
        return X


class model (BaseEstimator):
    def __init__(self, classifier=RandomForestClassifier()):
   
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
         Args :
            classifier : classifier we will use for making our predictions
            n_pca : argument for preprocessing
        '''
        self.is_trained=False # Etat de l'apprentissage
        self.classifier = classifier # Modèle de classification
        self.scoring_function = roc_auc_score # Méthode de calcul du score
        self.preprocess = preprocess()
        
    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        '''       
       
        self.preprocess.fit(X) # fit processing
        X = self.preprocess.transform(X) # transform
        self.classifier = self.load() # Rechargement du modèle optimisé
        self.classifier.fit(X, np.ravel(y)) # entrainement du modèle
        self.is_trained=True
        return self
   
    def predict(self, X):
        '''
        This function  provides predictions of labels on (test) data
        '''       
        X = self.preprocess.transform(X) # datatransform
        return self.classifier.predict(X) # make predictions
    
    def optimize(self, X, y, n_iter=100):
        """ 
        Optimise le classifieur en cherchant les meilleurs hyperparamètres
        Args:
            X : jeu de données d'entraînement
            y : labels correspondants
            n_iter : nombre de combinaisons testées (default=100)
        """
        # Paramètres à tester
        #print(self.classifier.get_params().keys())
        parameters={'bootstrap':[True,False],
                    'criterion':["gini", "entropy"], 
                    'n_estimators':[i for i in range(10,300,10)], 
                    'max_depth':[i for i in range(1,10)]+[None], 
                    'min_samples_split':[i for i in range(2,5)], 
                    'min_samples_leaf':[i for i in range(1,5)],
                    'random_state':[i for i in range(1,100)]}
        # Grille de recherche en utilisant toute la puissance processeur et paramétrée avec la fonction de score
        grid = RandomizedSearchCV(self.classifier, parameters, scoring=make_scorer(self.scoring_function), n_jobs=-1, n_iter=n_iter)
        print(grid.param_distributions)
        # Lancement des entrainements
        grid.fit(X, y)
        # Meilleurs hyperparamètres
        print(grid.best_estimator_)
        self.classifier = grid.best_estimator_
        
    def score(self, y_true, y_pred):
        """ 
        Calcul du score du modèle actuel 
        sur un jeu de données quelconque
        Args:
            y_true : labels réels
            y_pred : labels prédits
        """
        # Score du modèle
        return self.scoring_function(y_true, y_pred)
    
    def save(self, path="./"):
        pickle.dump(self.classifier, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        """ Rechargement du modèle préalablement enregistré """
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self.classifier = pickle.load(f)
                print("Model reloaded from: " + modelfile)
        return self.classifier
    
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