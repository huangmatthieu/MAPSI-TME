# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 10:47:08 2014

@author: huangmatthieu
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D

def bernoulli(p):
    return (np.random.rand() <= p)

def binomiale(n,p):
    sum = 0
    for i in range(n):
        sum += bernoulli(p)
    return sum

def galton(n):
#    vgalton = np.zeros((1,1000))[0,:]
    vgalton = np.arange(0,1000)
    for i in range(1000):
        vgalton[i] = binomiale(n,.5)
#        vgalton[binomiale(n,.5)] += 1
    return vgalton

#g = galton(20)
#plt.hist(g, 1000)
#plt.show()
#plt.clf()

####################################################################

def normale(k, sigma):
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
    x = np.linspace(-2*sigma, 2*sigma, k)
    y = (1/np.sqrt(2*np.pi) * sigma) * np.exp(-0.5 * (x*x)/(sigma*sigma))
#    y = np.exp(-x*x/(2.*sigma*sigma))/(np.sqrt(2. * np.pi) * sigma)
    return (x,y)

#x,y= normale(51, 5)
plt.plot(x, y)
#plt.show()
#plt.clf()


def proba_affine(k, slope):
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
    if abs(slope) > 2. / ( k * k ):
        raise ValueError ( 'la pente est trop raide : pente max = ' + 
        str ( 2. / ( k * k ) ) )
    x = np.arange(0,k)
    y = (1./k) + (x - ((k-1.)/2)) * slope
    return (x,y)
    
#x,y = proba_affine(11, .000002)
#plt.plot(x, y)
#plt.show()
#plt.clf()

def Pxy(Pa, Pb):
    MPa = np.zeros((len(Pa),1))
    MPa[:,0] = Pa   
    MPb = np.zeros((1,len(Pb)))
    MPb[0,:] = Pb
    return MPa.dot(MPb)
    
#PA = np.array ( [0.2, 0.7, 0.1] )
#PB = np.array ( [0.4, 0.4, 0.2] )
#print Pxy(PA, PB)

def dessine ( P_jointe ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace ( -3, 3, P_jointe.shape[0] )
    y = np.linspace ( -3, 3, P_jointe.shape[1] )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, P_jointe, rstride=1, cstride=1 )
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('P(A) * P(B)')
    plt.show ()

k = 21
Pa = normale(k, 2.5)[:,1]
Pb = proba_affine(k, 0.0045)[:,1]
#dessine(Pxy(Pa, Pb))


############################################################

def project1Var(P, index):
    """
    supprime 1 variable d'une probabilité jointe

    Param P : une distribution de proba jointe sous forme d'un array à 1 
       dimension ( toutes les variables aléatoires sont supposées binaires )
    Param index : représente la variable aléatoire à marginaliser 
       (0 = 1ère variable, 1 = 2ème variable, etc).
    """
    length = 2**( index + 1 )
    reste = 2**index
    vect = np.zeros ( P.size / 2 )
    for i in range ( P.size ):
        j = np.floor ( i / length ) * length / 2 + ( i % reste )
        vect[j] += P[i]
    return vect

def project(P, ind_to_remove):
    """
    Calcul de la projection d'une distribution de probas

    Param P une distribution de proba sous forme d'un array à 1 dimension
    Param ind_to_remove un array d'index representant les variables à
    supprimer. 0 = 1ère variable, 1 = 2ème var, etc.
    """
    v = P
    ind_to_remove.sort ()
    for i in range ( ind_to_remove.size - 1, -1, -1 ):
        v = project1Var ( v, ind_to_remove[i] )
    return v
    
def expanse1Var(P, index):
    """
    duplique une distribution de proba |X| fois, où X est une des variables
    aléatoires de la probabilité jointe P. Les variables étant supposées
    binaires, |X| = 2. La duplication se fait à l'index de la variable passé
    en argument.
    Par exemple, si P = [0,1,2,3] et index = 0, expanse1Var renverra
    [0,0,1,1,2,2,3,3]. Si index = 1, expanse1Var renverra [0,1,0,1,2,3,2,3].

    Param P : une distribution de proba sous forme d'un array à 1 dimension
    Param index : représente la variable à dupliquer (0 = 1ère variable,
       1 = 2ème variable, etc).
    """
    length = 2**(index+1)
    reste = 2**index
    vect = np.zeros ( P.size * 2 )
    for i in range ( vect.size ):
        j = np.floor ( i / length ) * length / 2 + ( i % reste )
        vect[i] = P[j]
    return vect
    
def expanse(P, ind_to_add):
    """
    Expansion d'une probabilité projetée

    Param P une distribution de proba sous forme d'un array à 1 dimension
    Param ind_to_add un array d'index representant les variables permettant
    de dupliquer la proba P. 0 = 1ère variable, 1 = 2ème var, etc.
    """
    v = P
    ind_to_add.sort ()
    for ind in ind_to_add:
        v = expanse1Var ( v, ind )
    return v

def nb_vars(P):
    i = P.size
    nb = 0
    while i > 1:
        i /= 2
        nb += 1
    return nb

def proba_conditionnelle(P):
    n = nb_vars(P) - 1
    Pb1 = project1Var(P, n)
    Pb = expanse1Var(Pb1, n)
    PA_B = P.copy()
    for i in range(len(P)):
        if (Pb[i] == 0):
            PA_B[i] = 0
        else:
            PA_B[i] /= Pb[i]
    return PA_B

P=np.array([0.06, 0.192, 0.06, 0.288, 0.14, 0.048, 0.14, 0.072])
P2 = proba_conditionnelle(P)
#print P2


asia = np.loadtxt("2014_tme2_asia.txt")
P2 = proba_conditionnelle(asia)

def is_indep(P, index, epsilon):
    n = nb_vars(P) - 1
    Pc = proba_conditionnelle(P)
    P2 = project1Var(P,index)
    Pc2 = proba_conditionnelle(P2)
    Pc2ex = expanse1Var(Pc2,index)   
    b = True
    for j in range(len(Pc)):
        if (abs(Pc[j] - Pc2ex[j]) > epsilon):
            b = False
    return b

def find_indep(P, epsilon):
    n = nb_vars(P) - 1                    # Nombre de variables dans B
    L = range(n, -1, -1)                  # Liste des variables dans B
    for i in range(n-1,-1,-1):
        if is_indep(P, i, epsilon):
            L.remove(i)
            P = project1Var(P, i)
    return (n+1, P, np.array(L))

R = find_indep(asia, 1e-6)
print R
Q = R[1]

print range(R[0]-1,-1,-1)
print range(R[0])

for i in range(R[0]):
    if not (i in R[2]):
        print i
        Q = expanse1Var(Q, i)
Pc = proba_conditionnelle(asia)

for i in range(len(Q)):
    if (abs(Q[i] - Pc[i]) > 1e-6):
        print i, Q[i], Pc[i]

print Pc.sum()
print Q.sum()

for i in range(8):
    print is_indep(asia, i, 1e-6)


Pasia = project1Var(asia, 6)
PPasia = expanse1Var(Pasia, 6)
L = np.array([0,1,2,3,4,5,7])
X=project(asia, L)
X=expanse(X,L)
PPasia *= X
for i in range(len(PPasia)):
    if (abs(PPasia[i] - asia[i]) > 5e-4):
        print i, PPasia[i], asia[i]

'''
LP = np.array([0,1])
LX = np.array([2,3,4,5,6,7])

Pasia = project(asia, LP)
PPasia = expanse(Pasia, LP)
X=project(asia, LX)
X=expanse(X,LX)
PPasia *= X
for i in range(len(PPasia)):
    if (abs(PPasia[i] - asia[i]) > 5e-3):
        print i, PPasia[i], asia[i]
        '''

'''
def find_all_indep(P, epsilon):
    n = nb_vars(P)
    R = []
    for i in range(n-1, 0, -1):
        R.append(find_indep(P, epsilon))
        P = project1Var(P, i-1)
    R.append(find_indep(P, epsilon))
    return R

R = find_all_indep(asia, 1e-6)
m = 0
for r in R:
    m += len(r[1])
print m
R.reverse()

#for r in R:
#    print r

P = R[0][1]
L = [0]
for i in range(1,len(R)):
    P = expanse1Var(P, i)
    Q = R[i][1]
    for j in range(0,i):
        if not (j in R[i][2]):
            Q = expanse1Var(Q, j)
    L.append(i)
    P *= Q

for i in range(0,len(P)):
    if (abs(P[i] - asia[i]) > 1e-6):
        print P[i], asia[i]
print P.sum()
print asia.sum()

print len(P)
'''
