# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 10:36:53 2014

@author: huangmatthieu
"""


import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
#import matplotlib.cm as cm

def tirage(m):
    return (2 * np.random.rand(2) - 1) * m

def monteCarlo(N):
    t = 2 * np.random.rand(N,2) - 1
    return 4 * np.where(t[:,0]**2 + t[:,1]**2 <= 1, 1, 0).mean(), t[:,0], t[:,1]

def tracerMonteCarlo():
    plt.figure()
    # trace le carré
    plt.plot([-1, -1, 1, 1], [-1, 1, 1, -1], '-')
    # trace le cercle
    x = np.linspace(-1, 1, 100)
    y = np.sqrt(1- x*x)
    plt.plot(x, y, 'b')
    plt.plot(x, -y, 'b')
    # estimation par Monte Carlo
    pi, x, y = monteCarlo(int(1e4))
    print "estimation:", pi
    # trace les points dans le cercle et hors du cercle
    dist = x*x + y*y 
    plt.plot(x[dist <=1], y[dist <=1], "go")
    plt.plot(x[dist>1], y[dist>1], "ro")
    plt.show()

def MCMC(N, m, burnt, mixt):
    x = 0
    y = 0
    i = 0
    j = 0
    k = 0
    accept = 1
    inCercle = 0
    while i < N:
        xt = x + np.random.rand() * 2 * m - m
        yt = y + np.random.rand() * 2 * m - m
        if (np.abs(xt) <= 1 and np.abs(yt) <= 1):
            x = xt
            y = yt
        if j > burnt:
            if k == mixt:
                k = 0
                i +=1
                if x**2 + y**2 <= 1:
                    inCercle +=1
            else:
                k += 1
        j += 1
    return 4. * inCercle / i

def swapF(f, chars):
    f2 = dict(f)
    l = len(f)
    a = np.random.randint(0, l)
    b = np.random.randint(0, l)
    t = f2[chars[a]]
    f2[chars[a]] = f2[chars[b]]
    f2[chars[b]] = t
    return f2

def decrypt(m, f):
    return u"".join([f[c] for c in m])

def logLikelihood(m, mu, A, chars):
    logL = np.log(mu[chars.index(m[0])])
    for i in range(len(m) - 1):
        logL += np.log(A[chars.index(m[i]),chars.index(m[i+1])])
    return logL

def MetropolisHastings(m, mu, A, f, N):
    maxd = decrypt(m, f)
    print maxd
    maxL = logLikelihood(maxd, mu, A, chars)
    logL = maxL
    while(N >= 0):
        N -= 1
        f2 = swapF(f, chars)
        d2 = decrypt(m,f2)
        logL2 = logLikelihood(d2, mu, A, chars)
        r = np.random.rand()
        logp = logL2 - logL
        if (r == 0 or np.log(r) <= logp):
            logL = logL2
            f = f2
            if (logL > maxL):
                maxL = logL
                maxd = d2
                print maxd
    return maxd

import numpy.random as npr

def updateOccurrences(text, count):
   for c in text:
      if c == u'\n':
         continue
      try:
         count[c] += 1
      except KeyError as e:
         count[c] = 1

def mostFrequent(count):
   bestK = []
   bestN = -1
   for k in count.keys():
      if (count[k]>bestN):
         bestK = [k]
         bestN = count[k]
      elif (count[k]==bestN):
         bestK.append(k)
   return bestK

def identityF(keys):
   f = {}
   for k in keys:
      f[k] = k
   return f

def replaceF(f, kM, k):
   try:
      for c in f.keys():
         if f[c] == k:
            f[c] = f[kM]
            f[kM] = k
            return
   except KeyError as e:
      f[kM] = k

def mostFrequentF(message, count1, f={}):
   count = dict(count1)
   countM = {}
   updateOccurrences(message, countM)
   while len(countM) > 0:
      bestKM = mostFrequent(countM)
      bestK = mostFrequent(count)
      if len(bestKM)==1:
         kM = bestKM[0]
      else:
         kM = bestKM[npr.random_integers(0, len(bestKM)-1)]
      if len(bestK)==1:
         k = bestK[0]
      else:
         k = bestK[npr.random_integers(0, len(bestK)-1)]
      replaceF(f, kM, k) 
      countM.pop(kM)
      count.pop(k)
   return f

#tracerMonteCarlo()
#print MCMC(1000, 1, 100, 100)

(count, mu, A) = pkl.load(file("countWar.pkl", "rb"))
secret = (open("secret2.txt", "r")).read()[0:-1] # -1 pour supprimer le saut de ligne

chars = count.keys()
fInit = identityF(count.keys())
fInit = mostFrequentF(secret, count, fInit)
MetropolisHastings(secret, mu, A, fInit, int(5e4))


