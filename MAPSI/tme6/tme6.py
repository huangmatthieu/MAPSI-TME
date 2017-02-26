# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:19:17 2014

@author: huangmatthieu
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

data = pkl.load(file("TME6_lettres.pkl","rb"))
X = np.array(data.get('letters')) # récupération des données sur les lettres
Y = np.array(data.get('labels')) # récupération des étiquettes associées 

# affichage d'une lettre
def tracerLettre(let):
    a = -let*np.pi/180; # conversion en rad
    coord = np.array([[0, 0]]); # point initial
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.figure()
    plt.plot(coord[:,0],coord[:,1])
    plt.savefig("exlettre.png")
    return
    
def discretise(X,d):
    intervalle=360./d
    Xd=[]
    for i in range(len(X)):
        res=np.floor(X[i]/intervalle)
        Xd.append(res)
    return np.array(Xd)
    
def groupByLabel( y):
    index = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        index.append(ind)
    return index
    
"""  
#initialisation des matrice A et Pi à zero

def learnMarkovModel(Xc, d):
    A = np.zeros((d,d))
    Pi = np.zeros(d)
    for i in range(len(Xc)):
        x = Xc[i]
        Pi[x[0]] += 1
        for j in range(len(x) - 1):
            A[x[j]][x[j+1]] += 1
    A = A/np.maximum(A.sum(1).reshape(d,1),1) # normalisation
    Pi = Pi/Pi.sum()
    return (Pi,A)
"""   

#initialisation des matrice A et Pi à un

def learnMarkovModel(Xc, d):
    A = np.ones((d,d))
    Pi = np.ones(d)
    for i in range(len(Xc)):
        x = Xc[i]
        Pi[x[0]] += 1
        for j in range(len(x) - 1):
            A[x[j]][x[j+1]] += 1
    A = A/np.maximum(A.sum(1).reshape(d,1),1) # normalisation
    Pi = Pi/Pi.sum()
    return (Pi,A)
    
def probaSequence(s, Pi, A):
    p = np.log(Pi[s[0]])
    for i in range(len(s) - 1):
        p += np.log(A[s[i]][s[i+1]])
    return p
    
# separation app/test, pc=ratio de points en apprentissage
def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:np.floor(pc*n)])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest

"""
d=3     # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation
index = groupByLabel(Y)  # groupement des signaux par classe
models = []
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd[index[cl]], d))

tab=[]
for i in range(26):
    tab.append(probaSequence(Xd[0], models[i][0], models[i][1]))
    
"""

itrain,itest = separeTrainTest(Y,0.8)

ia = []
for i in itrain:
    ia += i.tolist()    
it = []
for i in itest:
    it += i.tolist()

d=10 # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation
index = groupByLabel(Y)  # groupement des signaux par classe
models = []
"""
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd[index[cl]], d))
"""
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd[itrain[cl]], d))

proba = np.array([[probaSequence(Xd[i], models[cl][0], models[cl][1]) for i in it]for cl in range(len(np.unique(Y)))])

Ynum = np.zeros(Y.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num
    
pred = proba.argmax(0) # max colonne par colproba = np.array([[probaSeq(Xd[i], models[cl][0], models[cl][1]) for i in range(len(Xd))]for cl in range(len(np.unique(Y)))])onne

print np.where(pred != Ynum[it], 0.,1.).mean()


conf = np.zeros((26,26))
for k in range(len(it)):
    conf[pred[k]][Ynum[it[k]]] += 1

plt.figure()
plt.imshow(conf, interpolation='nearest')
plt.colorbar()
plt.xticks(np.arange(26),np.unique(Y))
plt.yticks(np.arange(26),np.unique(Y))
plt.xlabel(u'Vérité terrain')
plt.ylabel(u'Prédiction')
plt.savefig("mat_conf_lettres.png")

pi = models[0][0]
A = models[0][1]
picumule = np.cumsum(pi)
Acumule =[]
for i in range(len(A)):
    Acumule.append(np.cumsum(A[i]))

print pi
print picumule  
print A
print Acumule

""""
campagnes d'expérience
"""

itrain,itest = separeTrainTest(Y,0.8)
index = groupByLabel(Y)  # groupement des signaux par classe

ia = []
for i in itrain:
    ia += i.tolist()    
it = []
for i in itest:
    it += i.tolist()

Ynum = np.zeros(Y.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num

Nmax=30

#performance en test
for t in range(3,Nmax+1):
    Xd=discretise(X,t)
    models=[]
    for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
        models.append(learnMarkovModel(Xd[itrain[cl]], t))
    proba = np.array([[probaSequence(Xd[i], models[cl][0], models[cl][1]) for i in it]for cl in range(len(np.unique(Y)))])
    pred = proba.argmax(0) # max colonne par colproba = np.array([[probaSeq(Xd[i], models[cl][0], models[cl][1]) for i in range(len(Xd))]for cl in range(len(np.unique(Y)))])onne
    res=np.where(pred != Ynum[it], 0.,1.).mean()
    print "probablité des lettres reconnues pour une discrétisation sur "+str(t)+" états en test: ",res

#performance en apprentissage
for t in range(3,Nmax+1):
    Xd=discretise(X,t)
    models=[]
    for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
        models.append(learnMarkovModel(Xd[itrain[cl]], t))
    proba = np.array([[probaSequence(Xd[i], models[cl][0], models[cl][1]) for i in ia]for cl in range(len(np.unique(Y)))])
    pred = proba.argmax(0) # max colonne par colproba = np.array([[probaSeq(Xd[i], models[cl][0], models[cl][1]) for i in range(len(Xd))]for cl in range(len(np.unique(Y)))])onne
    res=np.where(pred != Ynum[ia], 0.,1.).mean()
    print "probablité des lettres reconnues pour une discrétisation sur "+str(t)+" états en apprentissage: ",res