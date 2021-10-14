# -*- coding: utf-8 -*-

"""
This file contains the library of functions for the "Wine Classification" lecture
"""

# Built-in/Generic Imports

# Libs
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
from scipy.special import expit 
import scipy.optimize as opt
import time
import matplotlib.patches as mpatches

## Data separation 
from sklearn.model_selection import train_test_split

## Normalization 
from sklearn import preprocessing

## learning curve
from sklearn.model_selection import learning_curve

## Supervised learning classification models from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

## Statistiques
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import make_scorer 

#from show import show

# Own modules

__author__ = 'Thierry SANDOZ and Stephen MONNET'
__copyright__ = 'Copyright 2020, Machine Learning HEIG-VD'
__version__ = '1.0.0'
__maintainer__ = 'Thierry SANDOZ and Stephen MONNET'
__email__ = 'thierry.sandoz@heig-vd.ch, stephen.monnet@heig-vd.ch'
__status__ = '{dev_status}'

#######################################################################################
# Description :                                                                       #
#                                                                                     #
# Cette librairie contient toutes les fonctions utilisées pour réaliser le projet     #
# "Wine Classification" du cours "Machine Learning" (2020)                            #
# Voici une liste de ces différentes fonctions :                                      #
#                                                                                     #                        
#                                                                                     #
#######################################################################################

def distribution(data, feature_label, transformed = False):
    """Histogram"""
    sns.set()
    sns.set_style("whitegrid")
    # Create figure
    fig = plt.figure(figsize = (11,5));
    # Skewed feature plotting
    for i, feature in enumerate([feature_label]):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Total Number")
        ax.set_ylim((0, 1500))
        ax.set_yticks([0, 200, 400, 600, 800])
        ax.set_yticklabels([0, 200, 400, 600, 800, ">1000"])
        # Plot aesthetics
        if transformed:
            fig.suptitle("Log-transformed Distributions", \
                         fontsize = 16, y = 1.03)
        else:
            fig.suptitle("Skewed Distributions", \
                         fontsize = 16, y = 1.03)
            fig.tight_layout()
            plt.show()
            
#######################################################################################            

def nnGradientDescent(nn_params, input_layer_size, hidden_layer_size, \
    num_labels, X, y, Lambda, maxiter):
    """Calcul des valeurs de la learning curve"""
    
    # Initialization of the cost variable
    J_learn = np.zeros((maxiter, 1));

    for i in range(maxiter) :  
        
        # Arguments of the minimization function
        arguments = (input_layer_size , hidden_layer_size, num_labels, X, y, Lambda)
        
        # Training (1 iteration)
        res = opt.minimize(nnCostFunction, x0=nn_params, args=arguments, method="L-BFGS-B", options={'maxiter':1, 'disp':True}, jac=True)
        
        # Recover nn_params which contains Theta1 and Theta2
        nn_params = res['x']
        
        # Compute the cost function using nn_params returned by the optimization function
        J_learn[i], grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

    return J_learn, nn_params

#######################################################################################

def sigmoid(z):
    """Calcul de la fonction sigmoid"""
    g = np.zeros(z.shape)
    g = expit(z)

    return g

#######################################################################################

def sigmoidGradient(z):
    """Calcul du gradient de la fonction sigmoid"""
    g=1.0/(1.0+np.exp(-z))
    g=g*(1-g)
    
    return g

#######################################################################################

def displayData(X):
    """displays 2D data
      stored in X in a nice grid. It returns the figure handle h and the
      displayed array if requested."""

    # Compute rows, cols
    m, n = X.shape
    example_width = round(np.sqrt(n))
    example_height = (n / example_width)

    # Compute number of items to display
    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m / display_rows)

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                           pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in np.arange(display_rows):
        for i in np.arange(display_cols):
            if curr_ex > m:
                break
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, : ]))
            rows = [pad + j * (example_height + pad) + x for x in np.arange(example_height+1)]
            cols = [pad + i * (example_width + pad)  + x for x in np.arange(example_width+1)]
            display_array[min(rows):max(rows), min(cols):max(cols)] = X[curr_ex, :].reshape(example_height, example_width) / max_val
            curr_ex = curr_ex + 1
        if curr_ex > m:
            break

    # Display Image
    display_array = display_array.astype('float32')
    plt.imshow(display_array.T)
    plt.set_cmap('gray')
    # Do not show axis
    plt.axis('off')
    show()
    
    ###################################################################################################################

def predict(Theta1, Theta2, X):
    """Prédiction de la valeur de sortie en fonction des entrées et des paramètres Thetax trouvés par l'entrainement"""

    # If X is 1-D, we change it to 2-D
    if X.ndim == 1:
        X = np.reshape(X, (-1,X.shape[0]))

    # Length of the X vector
    m = X.shape[0]
    
    # Number of output labels
    num_labels = Theta2.shape[0]

    # Storage variable for the results
    p = np.zeros((m,1))

    h_1 = sigmoid(np.dot(np.column_stack((np.ones((m,1)), X )), Theta1.T))
    h_2 = sigmoid(np.dot(np.column_stack((np.ones((m,1)), h_1)), Theta2.T))

    # Return the max value case number
    p = np.argmax(h_2, axis=1)

    return p + 1

################################################################################################

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
	num_labels, X, y, lambda_reg):
    """Calcul de la fonction de coût et du gradient pour réseau de neurones"""
    
    # Récupère les vecteur Theta1, Theta2 depuis la variable nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1), order='F')
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1), order='F')

    # Utile pour la suite
    m = len(X)
             
    # Initialisation du coût à 0
    J = 0;
    
    # Initialisation des paramètes utiles au calcul du gradient
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # On ajoute une colonne de 1 au vecteur X --> Bias unit
    X = np.column_stack((np.ones((m,1)), X)) # = a_1

    # Calcul de sigmoid(a_1*Theta1')
    a_2 = sigmoid(np.dot(X,Theta1.T))

    # Ajoute une colonne de 1 au vecteur a_2 --> Bias unit
    a_2 = np.column_stack((np.ones((a_2.shape[0],1)), a_2))

    # Calcul de sigmoid(a_2*Theta2')
    a_3 = sigmoid(np.dot(a_2,Theta2.T))

    # Cost Function sans régularisation

    # sauvegarde des classes dans la variable "labels"
    labels = y
    
    # y devient une matrice de dimensions : Longueur de X fois nombre de labels 
    y = np.zeros((m,num_labels))
    
    # y est un vecteur contenant des 1 aux positions qui correspondent au numéro de la classe
    for i in range(m):
        y[i, labels[i]-1] = 1
    
    # Seconde variable du coût qui servira à accumuler les valeurs intermédiaires
    cost = 0
    
    # Calcul du coût sans vectorisation --> Accumulation des valeurs dans la variable "cost"
    for i in range(m):
        cost += np.sum( y[i] * np.log( a_3[i] ) + (1 - y[i]) * np.log( 1 - a_3[i] ) )
        
    # Calcul du coût sans régularisation    
    J = -(1.0/m)*cost

    # Cost Function avec régularisation
    
    # On ne régularise pas la première colonne (Bias unit) donc on l'enlève des vecteurs Thetax_sum

    Theta1_sum = np.sum(np.sum(Theta1[:,1:]**2))
    Theta2_sum = np.sum(np.sum(Theta2[:,1:]**2))

    J = J + ((lambda_reg/(2.0*m))*(Theta1_sum+Theta2_sum))

    # Algorithme "Backpropagation"

    # Initialisation des Deltax (Delta maj.) à 0 
    Delta_1 = 0
    Delta_2 = 0

    # Pour chaque exemple d'entrainement
    for t in range(m):

        # Intialisae x à chaque valeur des exemples d'entrainement
        x = X[t]

        # Calcul de sigmoid(a_1*Theta1')
        a_2 = sigmoid(np.dot(x,Theta1.T))
        
        # Ajoute une colonne de 1 au vecteur a_2 --> Bias unit
        a_2 = np.concatenate((np.array([1]), a_2))

        # Calcul de sigmoid(a_2*Theta2')
        a_3 = sigmoid(np.dot(a_2,Theta2.T))

        # Initialisation de delta_3
        delta_3 = np.zeros((num_labels))

        # y_j indique si l'exemple appartient à la classe (y_j = 1) ou a une autre (y_j = 0)
        for j in range(num_labels):
            y_j = y[t, j]
            delta_3[j] = a_3[j] - y_j

        # Calcul de delta2 = Theta2' * delta3 .* sigmoidGradient(z2)
        delta_2 = (np.dot(Theta2[:,1:].T, delta_3).T) * sigmoidGradient( np.dot(x, Theta1.T))

        # Produit "dyadique" Source : https://fr.wikipedia.org/wiki/Produit_dyadique
        Delta_1 += np.outer(delta_2, x)
        Delta_2 += np.outer(delta_3, a_2)

    # On obtient le gradient en divisant Deltax par m (La longueur du vecteur X)
    Theta1_grad = Delta_1 / m
    Theta2_grad = Delta_2 / m

    # Régularisation pour le gradient
    
    # Copie de Theta1_grad dans une variable
    Theta1_grad_unreg = np.copy(Theta1_grad)
    Theta2_grad_unreg = np.copy(Theta2_grad)
    
    # Régularisation
    Theta1_grad += (float(lambda_reg)/m)*Theta1
    Theta2_grad += (float(lambda_reg)/m)*Theta2
    
    # Remise des valeurs non régularisée dans le vecteur de gradient
    Theta1_grad[:,0] = Theta1_grad_unreg[:,0]
    Theta2_grad[:,0] = Theta2_grad_unreg[:,0]

    # Concaténage des valeurs de gradien dans une seule variable "grad"
    grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))

    return J, grad

###############################################################################

def correlationMatrix(data) :
    correlation = data.corr()
    # display(correlation)
    plt.figure(figsize=(14, 12))
    heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
    plt.title('Matrice de corrélation')
    plt.show()
    
###############################################################################
    
def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])

###############################################################################
    
def histogram(data, bins, title) :
    
    plt.figure(figsize = (8, 5))
    plt.hist(data, bins=bins)
    bins_labels(bins, fontsize=20)
    plt.title(title)
    plt.show()
    
###############################################################################

def sigmoid(z) :
    g = 1 / (1 + np.exp(-z))
    return g
    

###############################################################################

def costFunctionLR(theta, X, y, lambda_reg) :

    m, n = X.shape;

    J = 0;

    i = np.arange(1, len(theta));
    h = sigmoid(np.dot(X,theta));

    J = np.divide(np.dot((-y).T, np.log(h)) - np.dot((1-y).T, np.log(1-h)), m) + lambda_reg*np.sum(theta[i]**2)/(2*m)
    
    return J

###############################################################################

def gradFunctionLR(theta, X, y, lambda_reg) :
    
    m, n = X.shape;

    grad = np.zeros(len(theta));
    i = np.arange(1, len(theta));
    h = sigmoid(np.dot(X,theta));
    k = np.divide(np.dot(X.T, (h-y)), m);
    grad[0] = -(grad[0] - k[0]);
    grad[i] = -(grad[i] - theta[i]*(lambda_reg/m) - k[i]);
    grad = grad.flatten()
    
    return grad

###############################################################################
    
def learningCurveLR(X, y, X_val, y_val, lambda_reg) :

    m, n = X.shape;

    error_train = np.zeros((m, 1));
    error_val   = np.zeros((m, 1));
    
    X_train_1 = np.ones(shape=(X.shape[0], X.shape[1] + 1))
    X_train_1[:, 1:] = X

    ## Add a column od ones to the X matrix
    X_test_1 = np.ones(shape=(X_val.shape[0], X_val.shape[1] + 1))
    X_test_1[:, 1:] = X_val

    for i in range(m) :
        X_train = X[1:i,:];
        Y_train = y[1:i];
        initial_theta = np.zeros(X_train.shape[1])
        theta = opt.fmin_cg(costFunctionLR, initial_theta, gradFunctionLR, (X_train_1, y, lambda_reg), disp=0)
        error_train[i] = costFunctionLR(theta, X_train_1, y, lambda_reg);
        error_val[i] = costFunctionLR(theta, X_test_1, y_val, 0);

    return error_train, error_val

###################################################################################

def train_predict_evaluate(learner, sample_size, X_train, y_train, X_test, y_test) :
    
    results = {}
    
    start = time.time()
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size]) 
    end = time.time()
    
    results['train_time'] = end - start
    
    start = time.time()
    predictions_train = learner.predict(X_train[:300])
    predictions_test = learner.predict(X_test)
    end = time.time()
    
    results['pred_time'] = end - start
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta = 0.5, average = 'micro')
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = 0.5, average = 'micro')
    
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
    
    return results

#################################################################################################

def visualize_classification_performance(results) :
    sns.set()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(2, 3, figsize = (20,11))
    bar_width = 0.07
    colors = ["#0000ff", "#0055ff", "#00aaff", "#00ffff", "#00ffaa", "#00ff00", "#aaff00", "#ffff00", "#ffaa00", "#ff5500", "#ff0000"]
    
    for k, learner in enumerate(results.keys()) :
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3) :
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.4, 5])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
                
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    ax[0, 1].axhline(y = 1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = 1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = 1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = 1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))
    
    patches = []
    for i, learner in enumerate(results.keys()) :
        patches.append(mpatches.Patch(label = learner, color = colors[i]))
    plt.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    plt.savefig("performancesAlgorithms.svg")
    plt.show()

#####################################################################################################3

