# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 10:47:50 2014

@author: huangmatthieu
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
# truc pour un affichage plus convivial des matrices numpy
np.set_printoptions(precision=2, linewidth=320)
plt.close('all')

data = pkl.load(file("TME6_lettres.pkl","rb"))
X = np.array(data.get('letters'))
Y = np.array(data.get('labels'))


"""
def initGD(X,N):
      Xd=np.floor(np.linspace(0,N-.00000001,len(X)))
      return Xd
"""

def initGD(X,N):
    s=[]
    for etat in X:
        s.append( np.floor(np.linspace(0,N-0.00000001,len(etat))))
    return np.array(s)
      
def discretise(X,d):
    intervalle=360./d
    Xd=[]
    for i in range(len(X)):
        res=np.floor(X[i]/intervalle)
        Xd.append(res)
    return np.array(Xd)
     

     
""" 
X0 = [ 1,  9,  8,  8,  8,  8,  8,  9,  3,  4,  5,  6,  6,  6,  7,  7,  8,  9,  0,  0,  0,  1,  1]
"""

"""
for x, y in zip(Xd[Y=='a'], q[Y=='a']):
    print "tab x",x
    print "tab y",y
"""   
    
    
def learnHMM(allx, allS, N, K):
    A = np.ones((N,N))*1e-8
    B = np.ones((N,K))*1e-8
    Pi = np.zeros(N)
    for x, q in zip(allx, allS):
        Pi[q[0]] += 1
        B[q[0],x[0]] +=1
        for t in range(1, len(x)):
            A[q[t-1], q[t]] +=1
            B[q[t], x[t]] +=1
    Pi = Pi/Pi.sum()
    A = A / np.maximum(A.sum(1).reshape(N,1),1)
    B = B / np.maximum(B.sum(1).reshape(N,1),1)
    return (Pi, A, B)


def viterbi(xd, Pi, A, B):
    nbEtat = len(Pi)
    nbObs = len(xd)
    delta = np.zeros((nbEtat, nbObs))
    psy = np.zeros((nbEtat, nbObs))
    a = np.log(A)
    b = np.log(B)
    delta[:, 0] = np.log(Pi[:]) + np.log(B[:,xd[0]])
    psy[:, 0] = -1    
    for t in range(nbObs-1):
        for j in range(nbEtat):
            argm = (delta[:,t] + a[:,j]).argmax()
            psy[j,t+1] = argm
            delta[j, t+1] = delta[argm,t] + a[argm,j] + b[j, xd[t+1]]
    s = np.arange(nbObs)
    s[nbObs-1] = delta[:,nbObs-1].argmax()
    p = delta[s[nbObs-1],nbObs-1]
    for t in range(1,nbObs):
        s[nbObs-t-1] = psy[s[nbObs-t],nbObs-t]
    return s, p

   
def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:np.floor(pc*n)])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest


def baumwelch(Xd, iTrain,iTest, N, K):
    nCl = 26
    eps = 1e-4
    Pi = np.zeros((nCl, N))
    A = np.zeros((nCl, N, N))
    B = np.zeros((nCl, N, K))
    probaClasse = np.zeros(nCl)
    q = initGD(Xd, N)
    # échantillon d'apprentissage
    for cl in range(nCl): # parcours de toutes les classes  
        Pi[cl], A[cl], B[cl] = learnHMM(Xd[iTrain[cl]], q[iTrain[cl]], N, K)
    # évaluation des performances sur données de test
    #pour le calcul de logL_k
    for cl in range(nCl): # parcours de toutes les classes                
        for i in iTest[cl]: #parcours des lettres de la classe
            proba = 0
            s_est, p_est = viterbi(Xd[i], Pi[cl], A[cl], B[cl])
            #if p_est == -inf :
               # continue
            q[i] = s_est  #nouvel état calculé par viterbi
            proba += p_est
        Pi[cl], A[cl], B[cl] = learnHMM(Xd[iTest[cl]], q[iTest[cl]], N, K) #nouveau modèle d'apprentissage par viterbi 
        probaClasse[cl] = proba
    logL_k = probaClasse.sum()
    #pour le calcul de logL_k+1
    for cl in range(nCl): # parcours de toutes les classes
        for i in iTest[cl]:
            proba = 0
            s_est, p_est = viterbi(Xd[i], Pi[cl], A[cl], B[cl])
            #if p_est == -inf :
                #continue
            q[i] = s_est
            proba += p_est
        Pi[cl], A[cl], B[cl] = learnHMM(Xd[iTest[cl]], q[iTest[cl]], N, K)
        probaClasse[cl] = proba
    logL_kplus1 = probaClasse.sum()
    logL = [logL_k, logL_kplus1]
    niter = 2
    while (logL_k-logL_kplus1)/logL_k >= eps:
        for cl in range(nCl): # parcours de toutes les classes
            for i in iTest[cl]:
                proba = 0
                s_est, p_est = viterbi(Xd[i], Pi[cl], A[cl], B[cl])
                #if p_est == -inf :
                    #continue
                q[i] = s_est
                proba += p_est
        Pi[cl], A[cl], B[cl] = learnHMM(Xd[iTest[cl]], q[iTest[cl]], N, K)
        probaClasse[cl] = proba
        logL_k = logL_kplus1
        logL_kplus1 = probaClasse.sum()
        logL.append(logL_kplus1)
        niter += 1
    """
    print logL
    #plot logL
    plt.figure()
    plt.plot(range(niter), logL,'g--', label="vraisemblance")
    plt.legend(loc=0)
    plt.savefig("logL.png")
    """
    return Pi, A, B
    

# affichage d'une lettre (= vérification bon chargement)
def tracerLettre(let):
    a = -let*np.pi/180;
    coord = np.array([[0, 0]]);
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.plot(coord[:,0],coord[:,1])
    return


def generateHMM(Pic, Ac, Bc, n):
    N = len(Ac)
    K = len(Bc[0])
    r = np.random.rand(1,2*n)[0]
    emission=[]
    for ka in range(N):
        if r[0] <= Pic[ka]:
            etat = ka
            for kb in range(K):
                if r[1] <= Bc[etat, kb]:
                    emission.append(kb)
                    break
            break
    for i in range(2, 2*n, 2):
        for ka in range(N):
            if r[i] <= Ac[etat,ka]:
                etat = ka
                for kb in range(K):
                    if r[i+1] <= Bc[etat, kb]:
                        emission.append(kb)
                        break
                break
    return etat, emission
    
    
    
"""
TEST
"""
"""
K = 10 # discrétisation (=10 observations possibles)
N = 7  # 5 états possibles (de 0 à 4 en python) 
Xd=discretise(X,K)
q=initGD(Xd,N)

#Pi, A, B = learnHMM(Xd[Y=='a'],q[Y=='a'],N,K)
#s_est, p_est = viterbi(Xd[0], Pi, A, B)
itrain,itest = separeTrainTest(Y,0.8)

ia = []
for i in itrain:
    ia += i.tolist()    
it = []
for i in itest:
    it += i.tolist()

Pi,A,B = baumwelch(Xd, itrain,itest, N, K)  
models = [] 
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append((Pi[cl], A[cl], B[cl]))

proba = np.array([[viterbi(Xd[i], Pi[cl], A[cl], B[cl])[1] for i in it]for cl in xrange(26)])

Ynum = np.zeros(Y.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num

pred = proba.argmax(0) # max colonne par colonne

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
plt.savefig("mat_conf_lettres_30obs_5states.png")

#Trois lettres générées pour 5 classes (A -> E)
n = 3          # nb d'échantillon par classe
nClred = 5   # nb de classes à considérer
fig = plt.figure()
for cl in xrange(nClred):
    Pic = models[cl][0].cumsum() # calcul des sommes cumulées pour gagner du temps
    Ac = models[cl][1].cumsum(1)
    Bc = models[cl][2].cumsum(1)
    long = np.floor(np.array([len(x) for x in Xd[itrain[cl]]]).mean()) # longueur de seq. à générer = moyenne des observations
    for im in range(n):
        s,x = generateHMM(Pic, Ac, Bc, int(long))
        intervalle = 360./10  # pour passer des états => angles
        newa_continu = np.array([i*intervalle for i in x]) # conv int => double
        sfig = plt.subplot(nClred,n,im+n*cl+1)
        sfig.axes.get_xaxis().set_visible(False)
        sfig.axes.get_yaxis().set_visible(False)
        tracerLettre(newa_continu)
        
plt.savefig("lettres_hmm.png")
"""
    
"""""""""""""""""""""""
Campagnes d'experiences
"""""""""""""""""""""""

itrain,itest = separeTrainTest(Y,0.8)
ia = []
for i in itrain:
    ia += i.tolist()    
it = []
for i in itest:
    it += i.tolist()

Ynum = np.zeros(Y.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num

probas=[]
probasT=[]
Nmax=10
Kmax=20

# En test et apprentissage
for t in range(5,Nmax):
    for j in range(10,Kmax+1):
        Xd=discretise(X,j)
        q=initGD(Xd,t)
        Pi,A,B = baumwelch(Xd, itrain,itrain, t, j)  
        models = [] 
        for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
            models.append((Pi[cl], A[cl], B[cl]))
        proba = np.array([[viterbi(Xd[i], Pi[cl], A[cl], B[cl])[1] for i in range(len(Xd))]for cl in xrange(26)])
        pred = proba.argmax(0) # max colonne par colonne
        res=np.where(pred[it] != Ynum[it], 0.,1.).mean()
        res2=np.where(pred[ia] != Ynum[ia], 0.,1.).mean()
        print "Pour "+str(t)+" états et "+str(j)+" obervations en test : ",res
        print "Pour "+str(t)+" états et "+str(j)+" obervations en apprentissage : ",res2
        probasT.append(res)
        probas.append(res2)
        
       
probas=np.array(probas)
moy=probas.mean()
var=probas.std()
print "En apprentissage :\n moyenne : "+str(moy)+", variance : "+str(var)

probasT=np.array(probasT)
moyT=probasT.mean()
varT=probasT.std()
print "En test :\n moyenne : "+str(moyT)+", variance : "+str(varT)
