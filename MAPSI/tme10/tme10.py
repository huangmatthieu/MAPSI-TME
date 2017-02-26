# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 10:51:18 2014

@author: huangmatthieu
"""

import numpy as np
#import pickle as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.cm as cm

# Renvoie Y, X, Eps
def genDonneesJouets(N, sig, a, b, c):
    X = np.random.rand(N)
    Eps = np.random.randn(N) * sig
    return X, a*X + b + Eps

def estimationParamProba(X, Y):
    cov = np.cov([X, Y])
    a_est = cov[0,1] / cov[0,0]
    b_est = Y.mean() - X.mean() * cov[0,1] / cov[0,0]
    return a_est, b_est

def estimationMoindresCarres(X, Y):
    return np.linalg.solve(X.transpose().dot(X), X.transpose().dot(Y))

def descenteGradient(X, ylin):
    eps = 5e-3
    nIterations = 30
    w = np.zeros(X.shape[1]) #init a 0
    allw = [w.copy()]
    for i in xrange(nIterations):
        w -= eps * 2 * X.transpose().dot(X.dot(w) - ylin)
        allw.append(w.copy())
    return w, np.array(allw)

def afficheDescGradient(X, ylin, allD, D_star):
    ngrid = 20
    d1range = np.linspace(-0.5, 8, ngrid)
    d2range = np.linspace(-1.5, 1.5, ngrid)
    d1, d2 = np.meshgrid(d1range, d2range)
    cost = np.array([[np.log(((X.dot(np.array([d1i,d2j]))-ylin)**2).sum()) for d1i in d1range] for d2j in d2range])
    fig = plt.figure()
    plt.contour(d1, d2, cost)
    plt.scatter(D_star[0], D_star[1], c='r')
    plt.plot(allD[:,0], allD[:,1],'b+-', lw=2 )
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(d1, d2, cost, rstride = 1, cstride=1 )    
    plt.show()
    
def genDonneesJouetsQuad(N, sig, a, b, c):
    X = np.random.rand(N)
    Eps = np.random.randn(N) * sig
    return X, a * (X**2) + b * X + c + Eps

def estimationMoindresCarresQuad(X, Y):
    return np.linalg.solve(Xe.transpose().dot(Xe), Xe.transpose().dot(Y))
'''
a = 6.
b = -1.
c = 1
N = 100
sig = .4

#Donnees Jouets    
X, Y = genDonneesJouets(N, sig, a, b, c)
    #Approche Probabiliste
a_est, b_est =  estimationParamProba(X, Y)
T = np.array([0,1])
plt.figure()
plt.scatter(X, Y)
plt.plot(T, a_est * T + b_est, 'b')
    #Approche Moindres Carrees
Xc = np.hstack((X.reshape(N,1),np.ones((N,1))))
D_star = estimationMoindresCarres(Xc, Y)
plt.plot(T, D_star[0] * T + D_star[1], 'r:')
    #Approche Descente Gradient
D_est, allD = descenteGradient(Xc, Y)
plt.plot(T, D_est[0] * T + D_est[1], 'g--')
plt.show()
    #Affichage descente gradient
afficheDescGradient(Xc, Y, allD, D_star)

#Donnees Quadratiques Moindres Carres
Xquad, Yquad = genDonneesJouetsQuad(N, sig, a, b, c)
Xe = np.hstack( ((Xquad**2).reshape(N,1), Xquad.reshape(N,1), np.ones((N,1))) )
Dquad = estimationMoindresCarresQuad(Xe, Yquad)
T = np.arange(0,1.1,.1)
plt.figure()
plt.scatter(Xquad, Yquad)
plt.plot(T, Dquad[0]*(T**2) + Dquad[1]*T + Dquad[2], 'b')
plt.show()
'''
#Donnees Reelles
data = np.loadtxt("winequality-red.csv", delimiter=";", skiprows=1)
N,d = data.shape # extraction des dimensions
d -= 1
pcTrain  = 0.7 # 70% des données en apprentissage
allindex = np.random.permutation(N)
indTrain = allindex[:int(pcTrain*N)]
indTest = allindex[int(pcTrain*N):]
X = data[indTrain,:-1] # pas la dernière colonne (= note à prédire)
Y = data[indTrain,-1]  # dernière colonne (= note à prédire)
 # Echantillon de test (pour la validation des résultats)
XT = data[indTest,:-1] # pas la dernière colonne (= note à prédire)
YT = data[indTest,-1]  # dernière colonne (= note à prédire)

Xr = np.hstack((X, np.ones((len(X),1))))
XTr = np.hstack((XT, np.ones((len(XT),1))))
D = estimationMoindresCarres(Xr, Y)

print "Moyenne des erreurs en apprentissage:", ((Xr.dot(D) - Y)**2).mean()
print "Moyenne des erreurs en test:",((XTr.dot(D) - YT)**2).mean()
print ""
print "Taux de reconnaissance en apprentissage:", np.where(np.round(Xr.dot(D)) == np.round(Y),1,0).mean()
print "Taux de reconnaissance en test:", np.where((np.round(XTr.dot(D)) == np.round(YT)),1, 0).mean()

