# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 10:55:29 2014

@author: huangmatthieu
"""

"""
Exercice 1
"""

import numpy as np

# fonction pour transformer les données brutes en nombres de 0 à n-1
def translate_data ( data ):
    # création des structures de données à retourner
    nb_variables = data.shape[0]
    nb_observations = data.shape[1] - 1 # - nom variable
    res_data = np.zeros ( (nb_variables, nb_observations ), int )
    res_dico = np.empty ( nb_variables, dtype=object )
    # pour chaque variable, faire la traduction
    for i in range ( nb_variables ):
        res_dico[i] = {}
        index = 0
        for j in range ( 1, nb_observations + 1 ):
            # si l'observation n'existe pas dans le dictionnaire, la rajouter
            if data[i,j] not in res_dico[i]:
                res_dico[i].update ( { data[i,j] : index } )
                index += 1
            # rajouter la traduction dans le tableau de données à retourner
            res_data[i,j-1] = res_dico[i][data[i,j]]
    return ( res_data, res_dico )


# fonction pour lire les données de la base d'apprentissage
def read_csv ( filename ):
    data = np.loadtxt ( filename, delimiter=',', dtype='string' ).T
    names = data[:,0].copy ()
    data, dico = translate_data ( data )
    return names, data, dico
    
names,data,dico=read_csv("2014_tme5_asia.csv")
names,data,dico=read_csv("2014_tme5_agaricus_lepiota.csv")



"""
Exercice 2
"""

# etant donné une BD data et son dictionnaire, cette fonction crée le
# tableau de contingence de (x,y) | z
def create_contingency_table ( data, dico, x, y, z ):
    # détermination de la taille de z
    size_z = 1
    offset_z = np.zeros ( len ( z ) )
    j = 0
    for i in z:
        offset_z[j] = size_z      
        size_z *= len ( dico[i] )
        j += 1
    # création du tableau de contingence
    res = np.zeros ( size_z, dtype = object )
    # remplissage du tableau de contingence
    if size_z != 1:
        z_values = np.apply_along_axis ( lambda val_z : val_z.dot ( offset_z ),
                                         1, data[z,:].T )
        i = 0
        while i < size_z:
            indices, = np.where ( z_values == i )
            a,b,c = np.histogram2d ( data[x,indices], data[y,indices],
                                     bins = [ len ( dico[x] ), len (dico[y] ) ] )
            res[i] = ( indices.size, a )
            i += 1
    else:
        a,b,c = np.histogram2d ( data[x,:], data[y,:],
                                 bins = [ len ( dico[x] ), len (dico[y] ) ] )
        res[0] = ( data.shape[1], a )
    return res
    
    
def sufficient_statistics(data, dico, x, y, z):
    cont = create_contingency_table(data, dico, x, y, z)
    lz = len(cont)
    lx = len(cont[0][1])
    ly = len(cont[0][1][0])
    Nxz = np.zeros((lz, lx))
    Nyz = np.zeros((lz, ly))
    Nz = np.arange(0, lz)
    cpt=0
    #on stocke la somme de chaque ligne pour chaque Nz dans Nxz
    #on stocke la somme de chaque colonne pour chaque Nz dans Nyz
    for k in range(lz): 
        if cont[k][0] != 0:
            # compteur de Nz differents de 0
            cpt += 1
            Nxz[k] = cont[k][1].sum(1) #somme sur chaque ligne
            Nyz[k] = cont[k][1].sum(0) #somme sur chaque colonne
    chi=0
    for k in range(lz):
        if (cont[k][0] != 0):
            for i in range(lx):
                for j in range(ly):
                    n = Nxz[k,i] * Nyz[k,j] / cont[k][0]
                    if (n != 0):
                        chi += (cont[k][1][i,j] - n) * (cont[k][1][i,j] - n) / n
    dof=(lx - 1) * (ly - 1) * cpt
    return chi, dof
    
sufficient_statistics(data, dico, 1, 2, [3])
sufficient_statistics ( data, dico, 0,1,[2])
sufficient_statistics ( data, dico, 0,1,[2,3,4])
sufficient_statistics ( data, dico, 0,3,[3,4])
sufficient_statistics ( data, dico, 1,2,[3,4])


"""
Exercice 3 et 4
"""

import scipy.stats as stats

def indep_score(data, dico, x, y, z):
    cz = 1
    for k in z:
        cz *= len(dico[k])
    if (len(data[0]) < (5 * len(dico[x]) * len(dico[y]) * cz)):
        return (-1, 1)
    chi, dof = sufficient_statistics(data, dico, x, y, z)
    return stats.chi2.sf(chi, dof), dof
    
indep_score ( data, dico, 1,3,[])
indep_score ( data, dico, 1, 7, [])
indep_score ( data, dico, 0, 1,[2, 3])
indep_score ( data, dico, 1, 2,[3, 4])


"""
Exercice 5
"""

def best_candidate(data, dico, x, z, alpha):
    if (x == 0):
        return []
    ymin = 0
    pmin, dof = indep_score(data, dico, x, 0, z)
    for y in range(1,x):
        pvalue, dof = indep_score(data, dico, x, y, z)
        if pmin > pvalue:
            pmin = pvalue
            ymin = y
    if (pmin > alpha):
        return []
    return [ymin]
    
best_candidate ( data, dico, 1, [], 0.05 )
best_candidate ( data, dico, 4, [], 0.05 )
best_candidate ( data, dico, 4, [1], 0.05 )
best_candidate ( data, dico, 5, [], 0.05 )
best_candidate ( data, dico, 5, [6], 0.05 )
best_candidate ( data, dico, 5, [6,7], 0.05 )

"""
Exercice 6
"""

def create_parents(data, dico, x, alpha):
    z = []
    Ly = best_candidate(data, dico, x, z, alpha)
    while len(Ly) != 0:
        z += Ly
        Ly = best_candidate(data, dico, x, z, alpha)
    return z
    
    
create_parents ( data, dico, 1, 0.05 )
create_parents ( data, dico, 4, 0.05 )
create_parents ( data, dico, 5, 0.05 )
create_parents ( data, dico, 6, 0.05 )

"""
Exercice 7
"""


import pydot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def learn_BN_structure(data, dico, alpha):
    parents=[]
    for x in range(len(data)):
        parents.append(create_parents(data, dico, x, alpha))
    return parents
    
def display_BN (node_names, bn_struct, bn_name, style):
    graph = pydot.Dot( bn_name, graph_type='digraph')
    # création des noeuds du réseau
    for name in node_names:
        new_node = pydot.Node(name, 
                              style="filled",
                              fillcolor=style["bgcolor"],
                              fontcolor=style["fgcolor"])
        graph.add_node(new_node)
    # création des arcs
    for node in range ( len ( node_names ) ):
        parents = bn_struct[node]
        for par in parents:
            new_edge = pydot.Edge ( node_names[par], node_names[node] )
            graph.add_edge ( new_edge )
    # sauvegarde et affaichage
    outfile = bn_name + '.png'
    graph.write_png( outfile )
    img = mpimg.imread ( outfile )
    plt.imshow( img )
    

style = { "bgcolor" : "#6b85d1", "fgcolor" : "#FFFFFF" }
parents = learn_BN_structure(data, dico, 0.05)
print parents
display_BN(names, parents, "asia", style)
plt.show()


"""
Exercice 7 bis
"""

import pyAgrum as gum
import gumLib.notebook as gnb
from gumLib.pretty_print import pretty_cpt


def learn_parameters ( bn_struct, ficname ):
    # création du dag correspondant au bn_struct
    graphe = gum.DAG ()
    nodes = [ graphe.addNode () for i in range ( bn_struct.shape[0] ) ]
    for i in range ( bn_struct.shape[0] ):
        for parent in bn_struct[i]:
            graphe.addArc ( nodes[parent], nodes[i] )
    # appel au BNLearner pour apprendre les paramètres
    learner = gum.BNLearner ( ficname )
    learner.useScoreLog2Likelihood ()
    learner.useAprioriSmoothing ()
    return learner.learnParameters ( graphe )
    
# création du réseau bayésien à la aGrUM
bn = learn_parameters ( bn_struct, "2014_tme5_asia.csv" )

# affichage de sa taille
print bn

# récupération de la ''conditional probability table'' (CPT) et affichage de cette table
pretty_cpt ( bn.cpt ( bn.idFromName ( 'bronchitis?' ) ) )

# calcul de la marginale
proba = gnb.getPosterior ( bn, {}, 'bronchitis?' )

# affichage de la marginale
pretty_cpt ( proba )

gnb.showPosterior ( bn, {}, 'bronchitis?' ) 

gnb.showPosterior ( bn, {'smoking?': 'true', 'tuberculosis?' : 'false' }, 'bronchitis?' )