#!/usr/bin/python3

#Author  : Saroj Kumar Dash
#Project : Chicago Crime Data Analysis

import copy
from path import Path
from crime_network import Crime_Network

class Normalize:
    def __init__(self):

        pass

    def maxMinNormalize(self, comm_something, idelta=0.0):
        """ Accept dictionary of dictionary structure """

        self.comm_star = copy.deepcopy (comm_something)
        self.size_comm_star = 0 #is computed in get_minmax
        imin,imax = self._get_minmax()

        if( idelta == 0.0):
            idelta = 1/self.size_comm_star
        for targets in self.comm_star:
            for target in (self.comm_star[targets]):
                currVal = self.comm_star[targets][target]
                if (imax == imin):
                    self.comm_star[targets][target] = idelta
                else:
                    self.comm_star[targets][target] = ((currVal - imin)/(imax-imin)) + idelta

        return self.comm_star
    
    def _get_minmax(self):
        comm_crime = self.comm_star
        allweights = []
        for targets in comm_crime:
            for target in (comm_crime[targets]):
                try:
                    allweights.append(float (comm_crime[targets][target]))
                except ValueError:
                    print ("Bad weight: ", comm_crime[targets][target])
                    allweights.append(0)

        self.size_comm_star = len(allweights)
        maxi = max(allweights)
        mini = min(allweights)

        return mini,maxi

if (__name__ == '__main__'):
    #client code starts here        
    path = Path ()


    ##Testing Crime network normalization
    c = Crime_Network (path.get_path (year=2011, month=1, type="crime"))
    comm_crime_network = c.get_network ()
    norm = Normalize()
    print(comm_crime_network[1])
    print ()
    norm_comm_crime = norm.maxMinNormalize(comm_crime_network)
    print(norm_comm_crime[1])


    ##Testing Police network normalization
    # p = Police_Network (path.get_path (year=2011, month=1, type="police"), -1, 30000)
    # comm_police_network = p.get_network ()
    # norm = Normalize(comm_police_network)
    # print(comm_police_network)
    # norm_comm_police = norm.maxMinNormalize()
    # print(norm_comm_police)


    ##Testing service community network normalization
    # types = ["sanity", "vehicles", "pot_holes", "lights_one", "lights_all", "lights_alley", "trees", "vacant"]
    # community = [-5, -6, -6, -5, -5, -5, -5, -5]
    # code = [40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000]

    # for i, name in enumerate (types):
    #     print ("\t\t- Adding {} network".format (name))
    #     s = ServiceNetwork (path.get_path (year=2011, month=1, type=name), community[i], code[i])
    #     comm_sanity = s.get_network ()
    # norm = Normalize(comm_sanity)
    # print(comm_sanity)
    # norm_comm_sanity = norm.maxMinNormalize()
    # print(norm_comm_sanity)
