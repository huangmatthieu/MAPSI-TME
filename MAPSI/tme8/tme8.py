# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:55:53 2014

@author: huangmatthieu
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as ml
import copy as cp


# fonction de suppression des 0 (certaines variances sont nulles car les pixels valent tous la même chose)
def woZeros(x):
    y = np.where(x==0., 1., x)
    return y

# Apprentissage d'un modèle naïf où chaque pixel est modélisé par une gaussienne (+hyp. d'indépendance des pixels)
# cette fonction donne 10 modèles (correspondant aux 10 classes de chiffres)
# USAGE: theta = learnGauss ( X,Y )
# theta[0] : modèle du premier chiffre,  theta[0][0] : vecteur des moyennes des pixels, theta[0][1] : vecteur des variances des pixels

def learnGauss (X,Y):
    theta = [(X[Y==y].mean(0),woZeros(X[Y==y].var(0))) for y in np.unique(Y)]
    return (np.array(theta))

    
def learnGaussGen (X,Y):
    theta = [(X[Y==y].mean(0),X[Y==y].var(0)) for y in np.unique(Y)]
    return (np.array(theta))

# Application de TOUS les modèles sur TOUTES les images: résultat = matrice (nbClasses x nbImages)
def logpobs(X, theta):
    logp = [[-0.5*np.log(mod[1,:] * (2 * np.pi )).sum() + -0.5 * ( ( x - mod[0,:] )**2 / mod[1,:] ).sum () for x in X] for mod in theta ]
    return np.array(logp)


def classifierBin(X, Y, eps,classe):
    d = len(X[0])
    Yc = np.where(Y==classe,1.,0.)
    w = (np.random.rand(d) - .5) * eps * 50
    b = (np.random.rand(1)[0] - .5) * eps * 50
    logL = []
    for i in range(2):
        fxi = (1./(1+np.exp(-(X.dot(w) + b))))
        logL.append((Yc * np.log(fxi) + (1 - Yc)*(np.log(1 - fxi))).sum())
        dbL = (Yc - fxi).sum()
        dwL = (X * (Yc - fxi).reshape(len(Yc), 1)).sum(0)
        b += eps * dbL
        w += eps * dwL
    k = 1
    seuil = 0.01
    while np.abs(1. * (logL[k-1] - logL[k]) / logL[k]) > seuil :
        fxi = (1./(1+np.exp(-(X.dot(w) + b))))
        logL.append((Yc * np.log(fxi) + (1 - Yc)*(np.log(1 - fxi))).sum())
        dbL = (Yc - fxi).sum()
        dwL = (X * (Yc - fxi).reshape(len(Yc), 1)).sum(0)
        b += eps * dbL
        w += eps * dwL
        k += 1
    return w, b, logL
    
def learnRL(X, Y, XT, YT, eps):
    thetaRL = [(classifierBin(X, Y, eps,i)[0],classifierBin(X, Y, eps,i)[1]) for i in np.unique(Y)]
    return np.array(thetaRL)
    
def generatifGauss(X,Y):
    theta = learnGaussGen(X,Y)
    for i in range(len(np.unique(Y))):
        mu = theta[i,0]
        sigma2 = theta[i,1]
        x = np.random.randn(1,len(sigma2)) * np.sqrt(sigma2) + mu
        infZero = ml.find(x<0)
        x[0][infZero]=0
        plt.figure()
        plt.imshow(x.reshape(16,16), cmap = cm.Greys_r, interpolation='nearest')


######################################################################
#########################     script      ############################


# Données au format pickle: le fichier contient X, XT, Y et YT
# X et Y sont les données d'apprentissage; les matrices en T sont les données de test
data = pkl.load(file('usps_small.pkl','rb'))

X = data['X']
Y = data['Y']
XT = data['XT']
YT = data['YT']

theta = learnGauss ( X,Y ) # apprentissage

logp  = logpobs(X, theta)  # application des modèles sur les données d'apprentissage
logpT = logpobs(XT, theta) # application des modèles sur les données de test

ypred  = logp.argmax(0)    # indice de la plus grande proba (colonne par colonne) = prédiction
ypredT = logpT.argmax(0)

print "Taux bonne classification en apprentissage : ",np.where(ypred != Y, 0.,1.).mean()
print "Taux bonne classification en test : ",np.where(ypredT != YT, 0.,1.).mean()
        
"""
TEST
"""      

for i in range(10):
    w,b, logL = classifierBin(X, Y, 1e-5,i) #pour la classe i
    plt.figure()
    plt.plot(range(len(logL)), logL,'g--', label="vraisemblance")
    plt.legend(loc=0)
    plt.savefig("log_v_classe_"+str(i))


thetaRL = learnRL(X, Y, XT, YT, 1e-5)

# si vos paramètres w et b, correspondant à chaque classe, sont stockés sur les lignes de thetaRL... Alors:

pRL  = np.array([[1./(1+np.exp(-x.dot(mod[0]) - mod[1])) for x in X] for mod in thetaRL ])
pRLT = np.array([[1./(1+np.exp(-x.dot(mod[0]) - mod[1])) for x in XT] for mod in thetaRL ])
ypred  = pRL.argmax(0)
ypredT = pRLT.argmax(0)
print "Taux bonne classification en apprentissage : ",np.where(ypred != Y, 0.,1.).mean()
print "Taux bonne classification en test : ",np.where(ypredT != YT, 0.,1.).mean()


#Matrice de confusion
nCl = len(np.unique(Y))
conf = np.zeros((nCl,nCl))
for k in range(len(YT)):
    conf[ypredT[k], YT[k]] += 1
#Affichage de la matrice de confusion
plt.figure()
plt.imshow(conf, interpolation='nearest')
plt.colorbar()
plt.xticks(np.arange(10),np.unique(Y))
plt.yticks(np.arange(10),np.unique(Y))
plt.xlabel(u'Vérité terrain')
plt.ylabel(u'Prédiction')
plt.savefig("mat_conf_lettres.png")


# Pour afficher les vecteurs des moyennes des pixels pour les modèles 0 à 9

generatifGauss(X,Y)


# visualisation du vecteur w de la regression logistique
    
for i in range(10):
    plt.figure()
    plt.imshow(thetaRL[i][0].reshape(16,16), cmap = cm.Greys_r, interpolation='nearest')
    plt.show()
    plt.savefig("num_"+str(i)+".png")


# Parmi tous les p(yc=1|xi) aucun ne dépasse 0.5

train = pRL.max(0) < 0.5
test = pRLT.max(0) < 0.5
print "Pourcentage de chiffres ne ressemblant à rien : "
print "En aprentissage : ",np.where(train == True, 1,0).sum()/len(train)
print "En test :",np.where(test == True, 1,0).sum()/len(test)


# Parmi tous les p(yc=1|xi), le max et le second plus grand sont très proches
"""
# tres lent

firstmax=[]
secondmax=[]
prlbis=cp.copy(pRL)
for i in range(len(X)):
    firstmax.append(prlbis[:,i].argmax())
    prlbis[prlbis[:,i].argmax(),i]=0
    secondmax.append(prlbis[:,i].argmax())
firstmax=np.array(firstmax)
secondmax=np.array(secondmax)
"""
#train
prlbis=cp.copy(pRL)
firstmax=prlbis.argmax(0)
prlbis[firstmax,np.array(range(len(X)))]=0
secondmax=prlbis.argmax(0)
#test
prltbis=cp.copy(pRLT)
firstmaxT=prltbis.argmax(0)
prltbis[firstmaxT,np.array(range(len(XT)))]=0
secondmaxT=prltbis.argmax(0)

ecart = np.zeros(len(X))
ecartT = np.zeros(len(XT))

for i in range(len(X)):
    ecart[i] = pRL[firstmax[i], i] - pRL[secondmax[i] ,i]
    
for i in range(len(XT)):
    ecartT[i] = pRLT[firstmaxT[i], i] - pRLT[secondmaxT[i] ,i]
    
train2 = ecart > 0.03
test2 = ecartT > 0.03

print "Pourcentage de rejet avec un ecart de 0.03:"
print "En apprentissage:", np.where(train2 == True, 0.,1.).mean()
print "En test:", np.where(test2 == True, 0.,1.).mean()