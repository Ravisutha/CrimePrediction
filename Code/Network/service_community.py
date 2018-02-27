#!/usr/bin/python3
#Community and service network
#Author :   Ravisutha Sakrepatna Srinivasamurthy
#Project:   Analysis of Chicago Crime Data

import sys
import datetime
import csv
import numpy as np
from path import Path

class ServiceNetwork:

    """ Creates network of all 311 services """
    def __init__(self, path, column=1, code=10000, debug=False):
        """ Accepts the path of service data and community data. """

        #Get the path for the required datasets
        self.service_path = path[0];

        #Community data column in the dataset
        self.column = column

        #Unique code for this data
        self.code = code

        #Set debug mode
        self.debug = debug

        #Build service network
        self._service_network ()

        #Print if debug mode is ON
        self._my_print (self.get_network())

    def get_network (self):
        """Returns a dict containing edge list from community to service . """

        return (self.community)

    #Build service network
    def _service_network (self):
        """Connect crimes and community. """

        self.community = {}

        #Initialize the community dictionary
        for i in range (1, 78):
            self.community[i] = {}

        #Open service dataset
        with open (self.service_path, "rt") as f:

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

                #Add to the dictionary
                for community in communities.split (','):
                    try: 
                        community = int (float (str(community)))
                    except ValueError:
                        self._my_print ("Community: Value error ({}), row number = {}".format (community, i + 1))
                        continue

                    #Check if community is valid or not
                    if (community in self.community):
                        if (not (self.code in self.community[community])):
                            #Start with weight = 1
                            self.community[community][self.code] = 1
                        else:
                            #Increase the weight every time you come across the community
                            self.community[community][self.code] += 1
                    else:
                        #Community not valid
                        self._my_print ("Bad Data: 311 service with community {}, row number = {}".format (community, i + 1))
                i += 1

    def _my_print (self, arg):
        """ Prints only if the debug mode is True. """

        if (self.debug == True):
            print (arg) 

def main ():
    """ Main function. """

    if (len(sys.argv) > 1):
        if (sys.argv[1] == '1'):
            debug = True
    else:
        debug = False

    path = Path ()

    year = 2015
    community = [-5, -6, -6, -5, -5, -5, -5, -5]
    code = [40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000]
    p = ["sanity", "vehicles", "pot_holes", "lights_one", "lights_all", "lights_alley", "trees", "vacant"]

    for i, name in enumerate (p):
        if (debug == True):
            print (path.get_path (year, name))
            print ("community: {}".format (community[i]))
        a = ServiceNetwork (path.get_path (year, name), community[i], code[i], debug)

if (__name__=='__main__'):
    main ()
