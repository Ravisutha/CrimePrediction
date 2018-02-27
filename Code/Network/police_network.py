#!/usr/bin/python3
#Community and police station network
#Author :   Ravisutha Sakrepatna Srinivasamurthy
#Project:   Analysis of Chicago Crime Data

import datetime
import csv
import numpy as np

class Police_Network:
    def __init__(self, path, column=-1, code=30000):
        """ Accepts the path of police station data and community data. """

        #Get the path for the required datasets
        self.police_path = path[0];

        #Community column
        self.column = column

        #Code for police network
        self.code = code

        #Build police network
        self._police_network ()

    def get_network (self):
        """Returns a dict containing edge list from community to police station. """

        return (self.community)

    #Build police station network
    def _police_network (self):
        """Connect crimes and community. """

        self.community = {}

        #Initialize the community dictionary
        for i in range (1, 78):
            self.community[i] = {}

        #Open Police dataset
        with open (self.police_path, "rt") as f:

            reader = csv.reader (f)
            i = 0

            for row in reader:

                #Skip first line
                if (i == 0):
                    i += 1
                    continue

                #Read last column
                communities = row[self.column].replace (')', '')
                communities = communities.replace ('(', '')
                communities = communities.replace (' ', '')

                for community in communities.split (','):
                    community = int (community)
                    if (community in self.community):
                        self.community[community][i + self.code] = 1
                    else:
                        print ("Bad Data: Police station with community {}".format (community))
                i += 1

def main ():
    """ Main function. """
    path = []
    path.append ("../../Data/Static/Map_police_community.csv")

    a = Police_Network (path)

    print (a.get_network ())
if (__name__=='__main__'):
    main ()
