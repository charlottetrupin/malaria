'''Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''

import warnings
warnings.filterwarnings('ignore')
import random # Pour l'optimisation du preprocessing
import time # Temps d'exécution
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
    def __init__(self, n_pca=6, n_esti=5):
        ''' Classe pour le preprocessing
        '''
        self.is_trained = False # Etat de l'apprentissage
        self.n_pca = n_pca
        self.pca = PCA(n_components=n_pca, svd_solver="full") # Preprocessing
        self.estimator = IsolationForest(n_estimators=n_esti,n_jobs= -1) # outlier detection
        
        
        
    def fit(self, X):
        X = X[:,[0,5,7,14,15,16]] # features importantes
        self.pca.fit(X)
        self.estimator.fit(X)
        self.is_trained = True
        
    def transform(self, X, y, remove_outliers = True):
        """ Preprocessing du jeu de données X """
        X = X[:,[0,5,7,14,15,16]] # features importantes
        if remove_outliers :  
            liste = []
            for i in range(X.shape[0]):
                if self.estimator.predict(X)[i] != -1 :    
                    liste.append(i)			 #on enlève les outliers
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
        
        t0 = time.time()
        self.load() # Rechargement du modèle optimisé
        self.preprocess.fit(X) # fit processing
        X,y = self.preprocess.transform(X,y) # transform
        self.classifier.fit(X, np.ravel(y)) # entrainement du modèle
        self.is_trained=True
        print("time=%d sec" % (time.time() - t0))
        return self
   
    def predict(self, X):
        '''
        This function  provides predictions of labels on (test) data
        '''       
        y = []
        X,_ = self.preprocess.transform(X,y, remove_outliers = False) # datatransform
        return self.classifier.predict(X) # make predictions

    def optimize_preprocess(self, X_train, y_train, X_test, y_test, n_iter=60, verbose=False):
        """
        Optimise le preprocessing en recherchant les meilleurs paramètres
        Args:
            X_train, y_train : jeu de données entraînement
            X_test, y_test : jeu de données test
        """
        # Dictionnaire du résultat
        res = {"n_pca":1, "n_esti":1, "score":0.}
        
        n_pca_max = 6; n_esti_max = 10 # Bornes
        product = list() # Produit cartésien
        # Initialisation du produit
        for i in range(1, n_pca_max+1):
            for j in range(1, n_esti_max+1):
                product.append((i, j))
        
        # Vérification du nombre d'itérations
        if n_iter > len(product):
            raise ValueError("n_iter > "+len(product))
        
        # Boucle principale
        iter = 0; t0 = time.time()
        while(iter < n_iter):
            if verbose:
                print("n_iter=%d/%d" % (iter+1, n_iter))
            combination = random.choice(product) # Combinaison aléatoire
            product.remove(combination) # Suppression combinaison
            # Initialisation du preprocess selon combinaison
            self.preprocess = preprocess(n_pca=combination[0], n_esti=combination[1])
            self.preprocess.fit(X_train) # fit processing
            X_transformed, y_transformed = self.preprocess.transform(X_train, y_train) # transform
            # Entraînement du modèle par défaut
            self.classifier.fit(X_transformed, y_transformed)
            # Score obtenu sur l'ensemble de test
            X_transformed, _ = self.preprocess.transform(X_test, [], remove_outliers = False) # datatransform
            score = self.scoring_function(y_test, self.classifier.predict(X_transformed))
            # Mise à jour éventuelle de la meilleure combinaison
            if res["score"] < score:
                res["n_pca"] = combination[0]
                res["n_esti"] = combination [1]
                res["score"] = score
            if verbose:
                print("n_pca={0}, n_esti={1}, score={2}".format(combination[0], combination[1], score))
                print("time=%d sec" % (time.time() - t0))
            iter +=1
        
        print(res)
        self.preprocess = preprocess(n_pca=res["n_pca"], n_esti=res["n_esti"])
    
    def optimize_model(self, X, y, n_iter=100):
        """ 
        Optimise le classifieur en cherchant les meilleurs hyperparamètres
        Args:
            X : jeu de données d'entraînement
            y : labels correspondants
            n_iter : nombre de combinaisons testées (default=100)
        """
        self.load() # Rechargement du modèle (notamment l'optimisation du preprocessing)
        self.preprocess.fit(X) # fit processing
        X, y = self.preprocess.transform(X,y) # transform
        # Paramètres à tester
        #print(self.classifier.get_params().keys())
        parameters={'bootstrap':[True,False],
                    'criterion':["gini", "entropy"], 
                    'n_estimators':[i for i in range(10,300,10)], 
                    'max_depth':[i for i in range(1,10)]+[None], 
                    'min_samples_split':[i for i in range(2,5)], 
                    'min_samples_leaf':[i for i in range(1,5)],
                    'random_state':[i for i in range(1,100)]}
        #t0 = time.time()
        # Grille de recherche en utilisant toute la puissance processeur et paramétrée avec la fonction de score
        grid = RandomizedSearchCV(self.classifier, parameters, scoring=make_scorer(self.scoring_function), n_jobs=-1, n_iter=n_iter)
        print(grid.param_distributions)
        # Lancement des entrainements
        grid.fit(X, y)
        # Meilleurs hyperparamètres
        print(grid.best_estimator_)
        self.classifier = grid.best_estimator_
        #print("time=%d sec" % (time.time() - t0))
        
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
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        """ Rechargement du modèle préalablement enregistré """
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                res = pickle.load(f)
                self = res
                print("Model reloaded from: " + modelfile)
        return res
    
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
        sns.heatmap(confusion_matrix(y, self.predict(X)), annot=True, fmt='g', ax=ax)

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Parasitized', 'Uninfected']);
        ax.yaxis.set_ticklabels(['Parasitized', 'Uninfected']);
        plt.show()