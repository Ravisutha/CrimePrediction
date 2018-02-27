#!/usr/bin/python3

import csv
from similarity import FindSimilarity

class SimilarTree:

    class Node:
        """ A node has data and it connects to its children. """

        def __init__ (self, id):
            """ Prepare a node. """

            self.id = id
            self.similar = 1
            self.children = []

    def __init__ (self, sim, year=2015):
        """ Initialize the tree. Get similarity matrix. """

        self.sim = sim
        self.total_sim = {}
        self.visited = {}
        self.next_visit = {}
        self._get_sim_threshold ()

    def _get_sim_threshold (self, new=None):
        """ Insert new nodes. """

        self.similarity = {}
        self.children = {}

        for i in range (self.sim.shape[0]):
            self.similarity[i] = []
            self.children[i] = []
            self.visited[i] = 0 
            self.next_visit[i] = 0 

            for j in range (self.sim.shape[1]):
                if (self.sim[i, j] > 0.1):
                    self.similarity[i].append (self.sim[i, j])     
                    self.children[i].append (j)     

    def union (self, root):
        """ Recursive function to connect all nodes. """ 

        self.visited[root.id] = 1

        for i, child in enumerate (self.children[root.id]):
            if (child != root.id and self.visited[child] == 0):
                self.next_visit[child] = 1
            self.visited[child] = 1

        for i, child in enumerate (self.children[root.id]):
            if (self.visited[child] != 1 or self.next_visit[child] == 1):
                c = self.Node (child)
                #c.similar = root.similar * self.similarity[root.id][i] 
                c.similar = root.similar * 0.9 
                #c.similar = self.similarity[root.id][i] * 0.9
                self.total_sim[child] = c.similar
                root.children.append (c)
                self.visited[child] = 1
                self.next_visit[child] = 0

                self.union (c)

    def get_total_sim (self):
        """ Returns one clustered similarity. """
        
        return (self.total_sim)
        

def print_sim (root, one_similarity, second_time=False, old_dissimilar=[]):
    similar = []
    dissimilar = []
    if (second_time == False):
        for i in range (77):
            try:
                if (one_similarity[i] > 0.1):
                    similar.append ((i, one_similarity[i]))
                else:
                    dissimilar.append ([i])
            except KeyError:
                dissimilar.append ([i])
        return ([similar, dissimilar])
    else:
        for i in range (77):
            if ([i] not in old_dissimilar):
                continue
            try:
                if (one_similarity[i] > 0.3):
                    similar.append ((i, one_similarity[i]))
                else:
                    dissimilar.append ([i])
            except KeyError:
                dissimilar.append ([i])
        return ([similar, dissimilar])

for year in range (2011, 2016):
    sim = FindSimilarity(year)
    [out_sim, dummy] = sim.get_similarity ()
    total_similarity = []
    #for year in range(2011, 2016):
    x = SimilarTree (out_sim, year)
    root = x.Node (0)
    x.total_sim[0] = 1
    x.union (root)
    one_similarity = x.get_total_sim ()
    [similar, dissimilar] = print_sim (root, one_similarity)
    if (similar != []):
        total_similarity.append (similar)

    for i in dissimilar:
        #print ("Similar communities for {}:".format (i))
        x = SimilarTree (out_sim, year)
        root = x.Node (i[0])
        x.total_sim[i[0]] = 1
        x.union (root)
        one_similarity = x.get_total_sim ()
        [similar, dissimilar] = print_sim (root, one_similarity, second_time=True, old_dissimilar=dissimilar)
        if (similar != []):
            total_similarity.append (similar)
        #print (similar)
        #print ("Dissimilar communities:")
        #print (dissimilar)

    print (total_similarity)

    path =  "../../Data/Total_Data/Output/"
    with open(path + "similarity" + str(year) + ".csv", 'w') as f:
        writer = csv.writer(f,  quoting=csv.QUOTE_NONNUMERIC)
        for sim in total_similarity:
            writer.writerow(sim)
