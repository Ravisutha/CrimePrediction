#!/usr/bin/python3

#Author :   Ravisutha Sakrepatna Srinivasamurthy
#           Saroj Kumar Dash
#Project:   Analysis of Chicago Crime Data

import networkx as nx
import datetime
import csv
import numpy as np

class Crime_Network:
    def __init__(self, path):

        #Get the path for the required three data
        self.data_path = path[0];
        self.type_path = path[1]
        self.comm_path = path[2];
        
        #Build the crime type dictionary
        self.built_type ()

    def built_type (self):
        """Build a dictionary of crime types according to chicago police website."""

        with open(self.type_path, 'rt') as f:
            reader = csv.reader(f)

            self.a = {}
            i = 0
            j = 11
            for row in reader:
                if (i == 0):
                    i += 1
                    continue
                temp = row[0]
                if (temp[0] == '0'):
                    self.a[temp[1:]] = j
                else:
                    self.a[temp] = j
                j += 1

    def convert_date (self, t):
        """ Convert date and time into usable format. """

        [date, time, m] = t.split (" ")

        [month, date, year] = date.split ("/")

        month = int (month)
        date = int (date)
        year = int (year)

        d = datetime.date (year, month, date)
        weekday = d.weekday()

        [t, temp, temp] = time.split (":")
        t = int (t)

        if (m == "PM"):
            if (t >= 1 and t < 6):              #1PM - 5PM
                time = 2
            elif (t >= 6 and t <= 11):          #6PM - 11PM
                time = 3
            else:                               #12PM
                time = 1
        if (m == "AM"):
            if (t < 7):                            #1AM - 6AM
                time = 4            
            elif (t >= 7 and t <= 11):             #7AM - 11:59AM
                time = 1
            else:                                  #12AM
                time = 4

        return (time, weekday)

    def build_network (self):
        """Build a adjacency matrix using the crime data."""

        self.A = np.zeros((490, 490))
        with open (self.data_path, "rt") as f:
            reader = csv.reader (f)
            
            i = 0
            for row in reader:
                #Skip first line
                if (i == 0):
                    i += 1
                    continue
                
                #Get date information
                date = row[2]
                (t, w) = self.convert_date (date)
                t = t - 1
                w += 4

                #Crime Type
                temp = row[4]

                if (temp[0] == '0'):
                    if (not(temp[1:] in self.a)):
                        c_type = 412
                    else:
                        c_type = self.a[temp[1:]]
                else:
                    if (not(temp in self.a)):
                        c_type = 412
                    else:
                        c_type = self.a[temp]


                #Community Type
                commu = int (row[13]) + 412

                self.A[commu][t] += 1 
                self.A[commu][w] += 1 
                self.A[commu][c_type] += 1 

    #Create graph using numpy array
    def create_graph (self):
        """ Convert adjacency matrix to networkx graph. """

        G = nx.from_numpy_matrix (self.A)
        return (G)

    #Write to file and view in gephi
    def write_file (self, G):
        """ Write the network to a file. Helps in visualization. """
        nx.write_graphml(G,'new.graphml')

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

path = []
path.append ("/home/ravi/Network_Science/Project/Data/Crimes_2015.csv")
path.append ("/home/ravi/Network_Science/Project/Data/IUCR.csv")
path.append ("/home/ravi/Network_Science/Project/Data/community.csv")

a = Crime_Network (path)
a.build_network ()
G = a.create_graph()
a.add_attributes(G)
a.write_file (G)
