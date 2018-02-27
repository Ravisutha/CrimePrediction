#!/usr/bin/python3
#Community and school network
#Author :   Ravisutha Sakrepatna Srinivasamurthy
#Project:   Analysis of Chicago Crime Data

import datetime
import csv
import numpy as np

class SchoolNetwork:
    def __init__(self, path, code=110000, debug=False):
        """ Accepts the path of school data and community data. """

        #Get the path for the required datasets
        self.school_path = path[0];

        #Code number for this network
        self.code = code

        #Debug mode ON/OFF
        self.debug = debug

        #Build school network
        self._school_network ()

    def get_network (self):
        """Returns a dict containing edge list from community to school station. """

        return (self.community)

    #Build school network
    def _school_network (self):
        """Connect schools and community. """

        self.community = {}

        #Initialize the community dictionary
        for i in range (1, 78):
            self.community[i] = {}

        #Open School dataset
        with open (self.school_path, "rt") as f:

            reader = csv.reader (f)
            i = 0

            #Go through all rows
            for row in reader:

                #Skip first line
                if (i == 0):
                    i += 1
                    continue

                #Read last column
                communities = row[-1].replace (')', '')
                communities = communities.replace ('(', '')
                communities = communities.replace (' ', '')

                #For multiple communities, assign the school data 
                for community in communities.split (','):
                    if (community != '1.1'):
                        community = int (float (community))

                    if (community in self.community):
                        try: 
                            self.community[community][i + self.code] = float (row[12])
                        except ValueError:
                            self._my_print ("Bad Data")
                            continue
                    else:
                        self._my_print ("Bad Data: School with community {}".format (community))
                i += 1

    def _my_print (self, argv):
        if (self.debug == True):
            print (argv)

def main ():
    """ Main function."""

    path = []
    path.append ("../../Data/2015/school_2015.csv")

    a = SchoolNetwork (path, debug=True)
    print (a.get_network ())

if (__name__=='__main__'):
    main ()
