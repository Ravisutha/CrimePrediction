#!/usr/bin/python3

#Author :   Ravisutha Sakrepatna Srinivasamurthy
#Project:   Analysis of Chicago Crime Data

import networkx as nx
import datetime
import csv
import numpy as np
from crime_network import Crime_Network
from police_network import Police_Network
from service_community import ServiceNetwork
from community_libraries import Library_Network
from school_network import SchoolNetwork
from crime_police import CrimePoliceNetwork
from path import Path
from normalize_network import Normalize

class Build_Network:
    def __init__(self):
        """Initialize the build network by storing the given dictionary. 
        Example:
            net = Build_Network ()
            net.load_network ()
            G = net.get_network ()
        """
        self.new = 1
        return

    def add_net (self, comm_dict, color_attr=None, other_attr=None):
        """Initialize the build network by storing the given dictionary. """

        self.comm_dict = comm_dict
        self.create_graph (color_attr, other_attr)

    def _create_comm_nodes (self):
        """Create 77 community nodes. """

        for community in range (1, 78):
            self.G.add_node (community, color='green')
            self.G.nodes[community]['type'] = 'community'+ str (community)

    #Create graph using numpy array
    def create_graph (self, color_attr=None, other_attr=None, comm_network=True):
        """ Convert community dictionary to networkx graph. """

        #If the graph is created for the first time
        if (self.new == 1):
            self.G = nx.Graph ()
            if (comm_network == True):
                self._create_comm_nodes ()
            self.new = 0

        for source in self.comm_dict:
            for target in self.comm_dict[source]:
                if (target not in self.G.nodes):
                    self.G.add_node (target)
                if (color_attr != None):
                    self.G.nodes[target]['color'] = color_attr
                if (other_attr != None):
                    self.G.nodes[target][other_attr[0]] =  other_attr[1]

                try:
                    self.G.add_edge (source, target, weight=float(self.comm_dict[source][target]), color='blue')
                except ValueError:
                    self.G.add_edge (source, target, weight=0.0, color='blue')

        #print ("Your network is ready")
        return (self.G)

    #Write to file and view in gephi
    def write_file (self, path="new.graphml"):
        """ Write the network to a file. Helps in visualization. """

        nx.write_graphml(self.G, path[0])

    #Get the network
    def get_network (self):
        """ Returns the current network in form of Graph """
        return(self.G)

    #Get attributes
    def get_attributes (self):
        """ Returns attributes of a community. """
        return (self.attr)

    #Add atributes
    def add_attributes (self, G):
        """ Helper attributes for better visualization. """

        for i in range (np.shape(self.A)[0]):
            if (i < 4):
                G.nodes[i]["color"] = 'green'
            elif (i < 11):
                G.nodes[i]["color"] = 'blue'
            elif (i < 413):
                G.nodes[i]["color"] = 'red'
            else:
                G.nodes[i]["color"] = 'yellow'

    #Load network
    def load_network (self, year=2015, month=1, save=False, connect=False):
        """ Loads the network. """

        path = Path ()
        norm = Normalize ()
        self.attr = {}

        print ("\t1. Creating network")

        #Get community and crime dictionary (Code: 10000)
        print ("\t2. Adding Crime Network")
        a = Crime_Network (path.get_path (year=year, month=month, type="crime"), 10000)
        comm_crime = a.get_network ()
        self.attr["crime"] = comm_crime
        comm_crime = norm.maxMinNormalize (comm_crime)
        self.add_net (comm_crime, 'red', ('type', 'crime'))

        #Get community and police station dictionary (Code: 30000)
        print ("\t3. Adding Police Network")
        p = Police_Network (path.get_path (year=year, month=month, type="police"), -1, 30000)
        comm_police = p.get_network ()
        self.attr["police"] = comm_police
        comm_police = norm.maxMinNormalize(comm_police)
        self.add_net (comm_police, 'orange', ('type', 'police'))

        #Get community and 311 service dictionary (Code: 40000 - 110000)
        print ("\t4. Adding 311 service Networks")
        p = ["sanity", "vehicles", "pot_holes", "lights_one", "lights_all", "lights_alley", "trees", "vacant"]
        community = [-5, -6, -6, -5, -5, -5, -5, -5]
        code = [40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000]

        for i, name in enumerate (p):
            print ("\t\t- Adding {} network".format (name))
            s = ServiceNetwork (path.get_path (year=year, month=month, type=name), community[i], code[i])
            comm_sanity = s.get_network ()
            self.attr[name] = comm_sanity
            comm_sanity = norm.maxMinNormalize(comm_sanity)
            self.add_net (comm_sanity, 'brown', ('type', name))

        #Get community and library dictionary (Code: 120000)
        #print ("5. Adding Library Networks")
        #l = Library_Network (path.get_path (year=year, month=month, "library"), -1, 120000)
        #comm_library = l.get_network ()
        #net.add_net (comm_library, 'white', ('type', 'library'))

        #Get community and school dictionary (Code: 130000)
        print ("\t6. Adding School Network")
        s = SchoolNetwork (path.get_path (year=year, month=month, type="school"), 130000)
        comm_school = s.get_network ()
        self.attr["school"] = comm_school
        comm_school = norm.maxMinNormalize (comm_school)
        self.add_net (comm_school, 'violet', ('type', 'school'))

        #Connect police and crime network
        if (connect == True):
            print ("\t7. Connecting Police and Crime Network")
            pc = CrimePoliceNetwork (path.get_path (year=year, month=month, type="police_crime"))
            crime_police = pc.get_network ()
            crime_police = norm.maxMinNormalize (crime_police)
            self.add_net (crime_police)

        if (save == True):
            net.write_file(path.get_path (year=year, month=month, type="output"))

if (__name__=='__main__'):
    """ Build network """

    years = [2011, 2012, 2013, 2014, 2015]
    for year in years:
        for month in range (1, 13):
            print ("For year: {} and for the month {}".format (year, month))
            net = Build_Network ()

            #net.load_network (year=year, month=month, connect=True, save=True)
            net.load_network (year=year, month=month)
            net.get_network ()
            net.get_attributes ()
    #years = [2011, 2012, 2013, 2014, 2015]
#    years = [2015]
#
#    path = Path ()
#    for year in years:
#        print ("For year: {}".format (year), end=' ')
#        for month in range (1, 13):
#            print ("and month: {}".format (month))
#            print ("\t1. Creating network")
#            net = Build_Network ()
#
#            #Get community and crime dictionary (Code: 10000)
#            print ("\t2. Adding Crime Network")
#            a = Crime_Network (path.get_path (year=year, month=month, type="crime"))
#            comm_crime = a.get_network ()
#            net.add_net (comm_crime, 'red', ('type', 'crime'))
#
#            #Get community and police station dictionary (Code: 30000)
#            print ("\t3. Adding Police Network")
#            p = Police_Network (path.get_path (year=year, month=month, type="police"), -1, 30000)
#            comm_police = p.get_network ()
#            net.add_net (comm_police, 'orange', ('type', 'police'))
#
#            #Get community and 311 service dictionary (Code: 40000 - 110000)
#            print ("\t4. Adding 311 service Networks")
#            p = ["sanity", "vehicles", "pot_holes", "lights_one", "lights_all", "lights_alley", "trees", "vacant"]
#            community = [-5, -6, -6, -5, -5, -5, -5, -5]
#            code = [40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000]
#
#            for i, name in enumerate (p):
#                print ("\t\t- Adding {} network".format (name))
#                s = ServiceNetwork (path.get_path (year=year, month=month, type=name), community[i], code[i])
#                comm_sanity = s.get_network ()
#                net.add_net (comm_sanity, 'brown', ('type', name))
#
#            #Get community and library dictionary (Code: 120000)
#            #print ("5. Adding Library Networks")
#            #l = Library_Network (path.get_path (year=year, month=month, "library"), -1, 120000)
#            #comm_library = l.get_network ()
#            #net.add_net (comm_library, 'white', ('type', 'library'))
#
#            #Get community and school dictionary (Code: 130000)
#            print ("\t6. Adding School Network")
#            s = SchoolNetwork (path.get_path (year=year, month=month, type="school"), 130000)
#            comm_school = s.get_network ()
#            net.add_net (comm_school, 'violet', ('type', 'school'))
#
#            #Connect police and crime network
#            print ("\t7. Connecting Police and Crime Network")
#            pc = CrimePoliceNetwork (path.get_path (year=year, month=month, type="police_crime"))
#            crime_police = pc.get_network ()
#            net.add_net (crime_police)
#
#            print ("\t8. Writing to the file {}".format (path.get_path (year=year, month=month, type="output")))
#            #net.write_file(path.get_path (year=year, month=month, type="output"))
