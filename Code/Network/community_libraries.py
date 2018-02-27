#!/usr/bin/python3
#Community and library network
#Author :   Ravisutha Sakrepatna Srinivasamurthy
#Project:   Analysis of Chicago Crime Data

import datetime
import csv
import numpy as np

class Library_Network:
    def __init__(self, path, column=1, code=10000):
        """ Accepts the path of library station data and community data. """

        #Get the path for the required datasets
        self.library_path = path[0];

        #Community data column in the dataset
        self.column = column

        #Unique code for this data
        self.code = code

        #Build library network
        self._library_network ()

    def get_network (self):
        """Returns a dict containing edge list from community to library station. """

        return (self.community)

    #Build library network
    def _library_network (self):
        """Connect crimes and community. """

        self.community = {}

        #Initialize the community dictionary
        for i in range (1, 78):
            self.community[i] = {}

        #Open Library dataset
        with open (self.library_path, "rt") as f:

            reader = csv.reader (f)
            i = 0

            for row in reader:

                #Skip first line
                if (i == 0):
                    i += 1
                    continue

                #Read community column
                communities = row[self.column].replace (')', '')
                communities = communities.replace ('(', '')
                communities = communities.replace (' ', '')

                #Read weight column
                weight = row [-1].replace (')', '')
                weight = weight.replace (')', '')
                weight = weight.replace (' ', '')

                try:
                    weight = int (weight)
                except ValueError:
                    weight = 0
                    print ("Bad weight: row {}".format (i + 1))

                if (weight == -1):
                    weight = 0

                #Add to the dictionary
                for community in communities.split (','):
                    community = int (community)

                    #Check if community is valid or not
                    if (community in self.community):
                        if (not (self.code in self.community[community])):
                            #Start with weight = 1
                            self.community[community][self.code] = weight
                        else:
                            #Increase the weight every time you come across the community
                            self.community[community][self.code] += weight
                    else:
                        #Community not valid
                        print ("Bad Data: Library with community {}, row = {}".format (community, i + 1))
                i += 1

def main ():
    """ Main function. """
    path = []
    path.append ("../../Data/2015/map_libraries_visitors.csv")

    a = Library_Network (path, -14, 40000)
    print (a.get_network())

if (__name__=='__main__'):
    main ()
