# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 18:39:09 2014

@author: huangmatthieu
"""


import numpy as np

def read_file ( filename ):
    """
    Lit un fichier USPS et renvoie un tableau de tableaux d'images.
    Chaque image est un tableau de nombres réels.
    Chaque tableau d'images contient des images de la même classe.
    Ainsi, T = read_file ( "fichier" ) est tel que T[0] est le tableau
    des images de la classe 0, T[1] contient celui des images de la classe 1,
    et ainsi de suite.
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )    
    nb_classes, nb_features = [ int( x ) for x in infile.readline().split() ]
    # creation de la structure de données pour sauver les images :
    # c'est un tableau de listes (1 par classe)
    data = np.empty ( 10, dtype=object )  
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    filler( data, data )
    # lecture des images du fichier et tri, classe par classe
    for ligne in infile:
        champs = ligne.split ()
        if len ( champs ) == nb_features + 1:
            classe = int ( champs.pop ( 0 ) )
            data[classe].append ( map ( lambda x: float(x), champs ) )    
    infile.close ()
    # transformation des list en array
    output  = np.empty ( 10, dtype=object )
    filler2 = np.frompyfunc(lambda x: np.asarray (x), 1, 1)
    filler2 ( data, output )
    return output
    

import matplotlib.pyplot as plt

def display_image ( X ):
    """
    Etant donné un tableau de 256 flotants représentant une image de 16x16
    pixels, la fonction affiche cette image dans une fenêtre.
    """
    # on teste que le tableau contient bien 256 valeurs
    if X.size != 256:
        raise ValueError ( "Les images doivent être de 16x16 pixels" )
    # on crée une image pour imshow: chaque pixel est un tableau à 3 valeurs
    # (1 pour chaque canal R,G,B). Ces valeurs sont entre 0 et 1
    Y = X / X.max ()
    img = np.zeros ( ( Y.size, 3 ) )
    for i in range ( 3 ):
        img[:,i] = X
    # on indique que toutes les images sont de 16x16 pixels
    img.shape = (16,16,3)
    # affichage de l'image
    plt.imshow( img )
    plt.show ()
    
"""
Test exo 1
"""

I=read_file("2014_tme3_usps_train.txt")
display_image(I[5][10])

"""
Exercice 2
"""

def learnML_class_parameters(classe):
    taille=len(classe)
    mu=0.
    sig=0.
    for j in range(taille):
        mu+=classe[j]
    mu/=taille
    for j in range(taille):
        x=classe[j]-mu
        sig+=x*x
    sig /= taille
    return mu,sig
    
"""
Test exo 2
"""

moy,var=learnML_class_parameters(I[1])

"""
Exercice 3
"""

def learnML_all_parameters(trainData):
    parameters = []
    for d in trainData:
        parameters.append(learnML_class_parameters(d))
    return parameters
    
"""
Test exo 3
"""

l=learnML_all_parameters(I)

"""
Exercice 4
"""

def log_likelihoods(image, parameters):
    nbClass = len(parameters)
    r = []
    for j in range(nbClass):        
        l = 0.
        for i in range(256):
            if parameters[j][1][i] != 0:
                x = image[i] - parameters[j][0][i]
                l += - np.log(np.sqrt(2. * np.pi * parameters[j][1][i])) -1/2*((x*x)/parameters[j][1][i])
        r.append(l)
    return np.array(r)
    
"""
test exo 4
""""

J=read_file("2014_tme3_usps_test.txt")
tab = log_likelihoods(J[0][5],l)

"""
Exercice 5
"""

def classify_image(image, params):
    return log_likelihoods(image, params).argmax()


"""
test exo 5
"""

chiffre=classify_image(J[8][58],l)
print chiffre

"""
Exercice 6
"""

def classify_all_image(test_data, parameters):
    n = len(parameters)
    res = np.zeros((n,n), float)
    for i in range(len(test_data)):
        for img in test_data[i]:
            chiffre = classify_image(img, parameters)
            res[i,chiffre] += 1.
        res[i,:] /= len(test_data[i])
    return res

"""
test exo 6
"""

res = classify_all_image(J,l)

"""
Exercice 7
"""

from mpl_toolkits.mplot3d import Axes3D

def dessine ( classified_matrix ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = linspace ( 0, 9, 10 )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, classified_matrix, rstride = 1, cstride=1 )


dessine(res)
plt.show()
