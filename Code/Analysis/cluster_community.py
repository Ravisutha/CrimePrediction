#!/usr/bin/python3

import networkx as nx
import csv
import numpy as np
from similarity import FindSimilarity

class ClusterCoefficient:

    def __init__ (self, sim):
        """ Initialize. """
        
#        if (sim == []):
#            sim = FindSimilarity(year)
#            [self.sim, dummy] = sim.get_similarity ()
#        else:
        self.sim = sim
    
    def get_co_efficient (self):
        """ Returns Clustering co-efficient. """

        th = 0.1
        for i in range(np.shape (self.sim)[0]):
            for j in range (np.shape (self.sim)[1]):

                if (self.sim[i, j] > th):
                    self.sim[i, j] = 1
                else:
                    self.sim[i, j] = 0

        G=nx.Graph (self.sim)
        l = nx.clustering (G)
        return (l)

if (__name__ == '__main__'):
    c = ClusterCoefficient ()
    print (c.get_co_efficient ())
