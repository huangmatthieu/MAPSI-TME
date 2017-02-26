# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 19:29:45 2014

@author: huangmatthieu
"""

"""
Exercice 1
"""

import numpy as np
from math import *
from pylab import *

def read_file ( filename ):
    """
    Lit le fichier contenant les données du geyser Old Faithful
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )
    for ligne in infile:
        if ligne.find ( "eruptions waiting" ) != -1:
            break
    # ici, on a la liste des temps d'éruption et des délais d'irruptions
    data = []
    for ligne in infile:
        nb_ligne, eruption, waiting = [ float (x) for x in ligne.split () ]
        data.append ( eruption )
        data.append ( waiting )
    infile.close ()
    # transformation de la liste en tableau 2D
    data = np.asarray ( data )
    data.shape = (data.size / 2, 2 )
    return data
    
"""
test exo 1
"""

data = read_file("2014_tme4_faithful.txt")


"""
Exercice 2
"""

def normale_bidim (x,z,params):
    mu_x,mu_z,sig_x,sig_z,rho=params
    a=((x-mu_x)/sig_x)**2
    b=((z-mu_z)/sig_z)**2
    c=(2*rho*(x-mu_x)*(z-mu_z)) / (sig_x*sig_z)
    d=2*(1-rho**2)
    e=2*np.pi*sig_x*sig_z*np.sqrt(1-rho**2)
    return np.exp(-(a-c+b)/d)/e
    
import matplotlib.pyplot as plt

def dessine_1_normale ( params ):
    # récupération des paramètres
    mu_x, mu_z, sigma_x, sigma_z, rho = params
    # on détermine les coordonnées des coins de la figure
    x_min = mu_x - 2 * sigma_x
    x_max = mu_x + 2 * sigma_x
    z_min = mu_z - 2 * sigma_z
    z_max = mu_z + 2 * sigma_z
    # création de la grille
    x = np.linspace ( x_min, x_max, 100 )
    z = np.linspace ( z_min, z_max, 100 )
    X, Z = np.meshgrid(x, z)
    # calcul des normales
    norm = X.copy ()
    for i in range ( x.shape[0] ):
        for j in range ( z.shape[0] ):
            norm[i,j] = normale_bidim ( x[i], z[j], params )
    # affichage
    fig = plt.figure ()
    plt.contour ( X, Z, norm, cmap=cm.autumn )
    plt.show ()
    
"""
test exo 2
"""

params=np.array([0.1,0.2,0.3,0.4,0.5])
p=normale_bidim(1,2,params)
    
dessine_1_normale(params)

"""
Exercice 3
"""

def dessine_normales ( data, params, weights, bounds, ax ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]
    # on détermine les coordonnées des coins de la figure
    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]
    # création de la grille
    nb_x = nb_z = 100
    x = np.linspace ( x_min, x_max, nb_x )
    z = np.linspace ( z_min, z_max, nb_z )
    X, Z = np.meshgrid(x, z)
    # calcul des normales
    norm0 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
            norm0[j,i] = normale_bidim ( x[i], z[j], params[0] )# * weights[0]
    norm1 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
             norm1[j,i] = normale_bidim ( x[i], z[j], params[1] )# * weights[1]
    # affichages des normales et des points du dataset
    ax.contour ( X, Z, norm0, cmap=cm.winter, alpha = 0.5 )
    ax.contour ( X, Z, norm1, cmap=cm.autumn, alpha = 0.5 )
    for point in data:
        ax.plot ( point[0], point[1], 'k+' )


def find_bounds ( data, params ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]
    # calcul des coins
    x_min = min ( mu_x0 - 2 * sigma_x0, mu_x1 - 2 * sigma_x1, data[:,0].min() )
    x_max = max ( mu_x0 + 2 * sigma_x0, mu_x1 + 2 * sigma_x1, data[:,0].max() )
    z_min = min ( mu_z0 - 2 * sigma_z0, mu_z1 - 2 * sigma_z1, data[:,1].min() )
    z_max = max ( mu_z0 + 2 * sigma_z0, mu_z1 + 2 * sigma_z1, data[:,1].max() )
    return ( x_min, x_max, z_min, z_max )


# affichage des données : calcul des moyennes et variances des 2 colonnes
mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()

# les paramètres des 2 normales sont autour de ces moyennes
params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [0.4, 0.6] )
bounds = find_bounds ( data, params )

# affichage de la figure
fig = plt.figure ()
ax = fig.add_subplot(111)
dessine_normales ( data, params, weights, bounds, ax )
plt.show ()
    
"""
Exercice 4
"""

def Q_i(data, current_params, current_weights):
    n = len(data)
    Q=np.zeros((n,2))    
    Q[:,0] = normale_bidim(data[:,0], data[:,1], current_params[0]) * current_weights[0]
    Q[:,1] = normale_bidim(data[:,0], data[:,1], current_params[1]) * current_weights[1]
    s = Q[:,0] + Q[:,1]
    #normalisation
    Q /= s.reshape(n,1) 
    return Q
    
"""
test exo 4
"""

Q = Q_i(data,params,weights)
    
"""
Exercice 5
"""

def M_step(data, Q, current_params, current_weights):
    sQ0 = Q[:,0].sum()
    sQ1 = Q[:,1].sum()
    pi0 = sQ0 / (sQ0 + sQ1)
    pi1 = sQ1 / (sQ0 + sQ1)
    mu_x0 = (Q[:,0] * data[:,0]).sum() / sQ0
    mu_x1 = (Q[:,1] * data[:,0]).sum() / sQ1
    mu_z0 = (Q[:,0] * data[:,1]).sum() / sQ0
    mu_z1 = (Q[:,1] * data[:,1]).sum() / sQ1
    sigma_x0 = np.sqrt((Q[:,0] * ((data[:,0] - mu_x0) * (data[:,0] - mu_x0))).sum() / sQ0)
    sigma_x1 = np.sqrt((Q[:,1] * ((data[:,0] - mu_x1) * (data[:,0] - mu_x1))).sum() / sQ1)
    sigma_z0 = np.sqrt((Q[:,0] * ((data[:,1] - mu_z0) * (data[:,1] - mu_z0))).sum() / sQ0)
    sigma_z1 = np.sqrt((Q[:,1] * ((data[:,1] - mu_z1) * (data[:,1] - mu_z1))).sum() / sQ1)
    rho0 = ((Q[:,0] * ((data[:,0] - mu_x0) * (data[:,1] - mu_z0)) / (sigma_x0 * sigma_z0)).sum()) / sQ0
    rho1 = ((Q[:,1] * ((data[:,0] - mu_x1) * (data[:,1] - mu_z1)) / (sigma_x1 * sigma_z1)).sum()) / sQ1
    return (np.array([(mu_x0, mu_z0, sigma_x0, sigma_z0, rho0), 
                      (mu_x1, mu_z1, sigma_x1, sigma_z1, rho1)]),
            np.array([pi0, pi1]))
            
"""
test exo 5
"""

params,weights = M_step(data,Q,params,weights)
            
            
"""
Exercice 6
"""

def EM_1(data):
    L=[]
    l=[]
    mean1 = data[:,0].mean ()
    mean2 = data[:,1].mean ()
    std1  = data[:,0].std ()
    std2  = data[:,1].std ()
    params = np.array ( [(9,47,4,6,0.9),
                     (2,4,35,5,0.5)] )
    weights = np.array ( [ 0.3,0.7 ] )
    bounds = find_bounds ( data, params )
    for i in range(20):
        Q = Q_i(data, params, weights)
        sQ0 = Q[:,0].sum()
        sQ1 = Q[:,1].sum()
        logL = sQ0 * (np.log(weights[0]) - np.log(2*np.pi - np.log(params[0][2]) - np.log(params[0][3]) - .5 * np.log(1 - params[0][4])))
        + sQ1 * (np.log(weights[1]) - np.log(2*np.pi - np.log(params[1][2]) - np.log(params[1][3]) - .5 * np.log(1 - params[1][4])))
        L.append([i+1,logL])
        l.append(logL)
        params, weights = M_step(data, Q, params, weights)
    return L,l
        """
        fig = plt.figure()    
        ax = fig.add_subplot(111)
        dessine_normales( data, params, weights, bounds, ax )
        plt.show()
        """

"""
test exo 6
"""

#affiche 4 figure
l,L=EM_1(data)
print L
for i in range(1,20):
    if(abs(L[i]-L[i-1]))<0.01:
        conv=i-1
        break
 
print "convergence:",conv

"""
Exercice 7
"""

def EM(data):
    mean1 = data[:,0].mean ()
    mean2 = data[:,1].mean ()
    std1  = data[:,0].std ()
    std2  = data[:,1].std ()
    params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                         (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
    weights = np.array ( [ 0.5, 0.5 ] )
    res_EM = []
    for i in range(20):
        Q = Q_i(data, params, weights)
        params, weights = M_step(data, Q, params, weights)
        res_EM.append((params, weights))
    return res_EM
    
"""
test exo 7
"""

res_EM = EM(data)

# calcul des bornes pour contenir toutes les lois normales calculées
def find_video_bounds ( data, res_EM ):
    bounds = np.asarray ( find_bounds ( data, res_EM[0][0] ) )
    for param in res_EM:
        new_bound = find_bounds ( data, param[0] )
        for i in [0,2]:
            bounds[i] = min ( bounds[i], new_bound[i] )
        for i in [1,3]:
            bounds[i] = max ( bounds[i], new_bound[i] )
    return bounds

bounds = find_video_bounds ( data, res_EM )


import matplotlib.animation as animation

# création de l'animation : tout d'abord on crée la figure qui sera animée
fig = plt.figure ()
ax = fig.gca (xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]))

# la fonction appelée à chaque pas de temps pour créer l'animation
def animate ( i ):
    ax.cla ()
    dessine_normales (data, res_EM[i][0], res_EM[i][1], bounds, ax)
    ax.text(5, 40, 'step = ' + str ( i ))
    print "step animate = %d" % ( i )

# exécution de l'animation
anim = animation.FuncAnimation(fig, animate,
                               frames = len ( res_EM ), interval=500 )
plt.show ()

# éventuellement, sauver l'animation dans une vidéo
# anim.save('old_faithful.avi', bitrate=4000)