##############################################################################################
# Fichier contenant 2 classes pour le projet Malaria (Mini-Projet)                           # 
#                                                                                            #
# Contributeurs principaux :                                                                 #
# ------------------------                                                                   #
# 1. Partie Preprocessing -> Alicia Bec & Charlotte Trupin                                   #
# 2. Partie Model -> Maxime Vincent & Baptiste Maquet                                        #
# 3. Partie Visualization -> Sakina Atia & Mathilde Lasseigne                                #
#                                                                                            #
# Historique des modifications :                                                             #
# ----------------------------                                                               #
# 1. Suppression d'attributs de la classe model                                              #
# 2. Suppression de vérifications de méthodes de la classe model                             #
# 3. Ajout de méthodes dans la classe model :                                                #
#      3.1. Méthode model::score pour calculer le score                                      #
#      3.2. Méthode model::optimize_model pour optimiser les hyperparamètres                 #
#      3.3. Méthode model::confusion_matrix pour afficher une matrice de confusion           #
# 4. Ajout d'une classe preprocess au sein du fichier model.py                               #
# 5. Ajout d'une méthode model::optimize_preprocess pour tenter d'optimiser le preprocessing #
#                                                                                            #
# Date de dernière modification :                                                            #
# -----------------------------                                                              #
# https://github.com/charlottetrupin/malaria/commits/master/starting_kit/                    #
# sample_code_submission/model.py                                                            #
#                                                                                            #
##############################################################################################

import warnings
warnings.filterwarnings('ignore')
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
    def __init__(self, n_pca=9):
        ''' Classe pour le preprocessing
        '''
        self.is_trained = False # Etat de l'apprentissage
        self.n_pca = n_pca
        self.pca = PCA(n_components=n_pca) # Preprocessing
        self.estimator = IsolationForest(n_estimators=5,n_jobs= -1) # outlier detection



    def fit(self, X):
        X = X[:,[0,3,5,6,7,8,14,15,16]]
        self.pca.fit(X)
        self.estimator.fit(X)
        self.is_trained = True

    def transform(self, X, y, remove_outliers = True):
        """ Preprocessing du jeu de données X """
        X = X[:,[0,3,5,6,7,8,14,15,16]]
        if remove_outliers :
            liste = []
            for i in range(X.shape[0]):
                if self.estimator.predict(X)[i] != -1 :
                    liste.append(i)
            X = X[liste, :]
            y = y[liste]
        X = self.pca.transform(X) # reduce dimension
        return X,y



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
        X,y = self.preprocess.transform(X,y) # transform
        self.classifier = self.load() # Rechargement du modèle optimisé
        self.classifier.fit(X, np.ravel(y)) # entrainement du modèle
        self.is_trained=True
        return self

    def predict_proba(self, X):
        '''
        This function  provides predictions of labels on (test) data
        '''
        y = []
        X,_ = self.preprocess.transform(X,y, remove_outliers = False) # datatransform
        return self.classifier.predict_proba(X) # make predictions



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
        sns.heatmap(confusion_matrix(y, self.classifier.predict_proba(X)), annot=True, fmt='g', ax=ax)

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels');
        ax.set_title('Confusion Matrix');
        ax.xaxis.set_ticklabels(['Parasitized', 'Uninfected']);
        ax.yaxis.set_ticklabels(['Parasitized', 'Uninfected']);
        plt.show()
