#!/usr/bin/python3

import sys
import datetime
import csv
import networkx as nx
#import graphsim as gs
import numpy as np
import scipy as sp
sys.path.append('../Network/')
from make_network import Build_Network
from crime_network import Crime_Network
from police_network import Police_Network
from service_community import ServiceNetwork
from community_libraries import Library_Network
from school_network import SchoolNetwork
from path import Path

class FindSimilarity:
    """ Implementation of different similarity measures. """

    def __init__(self, year, month=1, load=True):
        """ Accept the graph, G """

        self.node_index = {}
        self.index_node = {}

        if (load == True):
            print ("Loading data")
            self.G = self.load_data (year, month)

    def _iterator_matrix(self, itr, m, save=False, path="sim.csv", norm=False):
        """ Converts a iterator to numpy matrix"""

        sim = np.zeros ((m, m))

        for i, j, k in itr:
            if (i > 77 or j > 77):
                continue
            if (norm == True):
                sim[i-1, j-1] = k/100
                sim[j-1, i-1] = k/100
            else:
                sim[i-1, j-1] = k
                sim[j-1, i-1] = k

        for i in range (m):
            sim [i - 1, i - 1] = 1

        if (save == True):
            with open(path, 'w') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                writer.writerows(sim)

        return (sim)

    def ascos_similarity (self):
        """ Uses graphsim node node similarity to find similarity between communities. """

        node_ids = self.G.nodes ()
        #sim = gs.ascos (self.G, is_weighted=True)
        return (sim)

    def get_community_nodes (self):
        """ Returns list of (u, v) from all communities to all communities. """

        #Nodes contain all possible (source, target) edges
        nodes  = []

        #Generate all possible combination of edges
        for i, s in enumerate (self.G.nodes ()):
            for j, t in enumerate (self.G.nodes ()):
                if (s > 77 or t > 77):
                    continue
                
                nodes.append ((s, t))
        return (nodes)

    def jaccard_similarity(self, only_community=True):
        """ Finds  the jacardian coefficient similarity"""

        nodes = self.get_community_nodes ()

        #Find jaccard similarity
        jacc_sim_itr = nx.jaccard_coefficient(self.G, nodes) 

        #Print similarity matrix
        return (self._iterator_matrix (jacc_sim_itr, 77, save=True, path="sim_jacard_2015.csv"))

    def adam_similarity (self):
        """ Finds adam similarity. """

        #Get communities nodes
        nodes = self.get_community_nodes ()

        #Find adam similarity
        adam_sim_itr = nx.adamic_adar_index(self.G) 

        #Return similarity matrix
        return (self._iterator_matrix (adam_sim_itr, 77, save=True, path="sim_adam.csv", norm=True))

    def pseudo_inverse_laplacian(self):
        """ Getting the pseudo inverse to obtain the random walk similarity"""

        nodes = self.get_community_nodes ()

        L = nx.laplacian_matrix(self.G)

        dense_L = L.todense()

        pinv_L = np.linalg.pinv(dense_L)
        
        inv_L = pinv_L.A
        
        inv_L = inv_L [0:77, 0:77]

        self._inverse_nomalize(inv_L)

        np.savetxt ("pseudo_inverse_laplacian.txt", inv_L)

        return (inv_L)

    def _inverse_nomalize (self, inv_L):
        """ Normalize the ouput from inverse laplacian matrix. """
        
        print ("Size of each row: {}".format (np.size(inv_L, axis=1)))
        for i in range(np.size(inv_L, axis=1)):
            d = inv_L[i, i]
            m = np.min(inv_L[i, :])
            
            for j in range (np.size(inv_L, axis=1)):
                inv_L[i, j] = inv_L[i, j] - m / d

    def load_data (self, year=2015, month=1):
        """ Load the chicago crime data and represent as network. """

        self.load = True

        path = Path ()
        net = Build_Network ()
        net.load_network (year=year, month=month)
        self.attr = net.get_attributes ()
        G = net.get_network ()

        return (G)

    def get_similarity (self, jaccard=True, r_walk=False, adam=False):
        """ Returns similarity for loaded network. """

        if (self.load == False):
            print ("Please load the network first. Ex: similarity.load_data ()")
            return (-1)

        if (jaccard == True or adam == True or r_walk == True):
            if (jaccard == True and adam == False and r_walk == False):
                return ([self.jaccard_similarity(), self.G])
            elif (jaccard == False and adam == True and r_walk == False):
                return ([self.adam_similarity(), self.G])
            elif (jaccard == False and adam == False and r_walk == True):
                return ([self.pseudo_inverse_laplacian (), self.G])
            else:
                return (self.G)

    def get_attributes (self):
        """ Returns attributes of the network. """

        if (self.load == False):
            print ("Please load the network first. Ex: similarity.load_data ()")
            return (-1)

        return (self.attr)

if (__name__ == '__main__'):

    sim = FindSimilarity(2015, month=1)
    sim.adam_similarity ()
    sim.jaccard_similarity ()
    sim.pseudo_inverse_laplacian ()
    #sim.pseudo_inverse_laplacian ()
    #attr = sim.get_attributes ()
