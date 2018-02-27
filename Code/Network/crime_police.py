#!/usr/bin/python3
#Aim    :   Connect crime and police data
#Author :   Ravisutha Sakrepatna Srinivasamurthy
#Project:   Analysis of Chicago Crime Data

import pandas as pd
from pandas import DataFrame, read_csv
import csv
from path import Path

class CrimePoliceNetwork:
    """Connects crime types to police stations."""

    def __init__(self, path, crime_code=10000, police_code=30000):
        """ Initializes crime police network.
        Parameters:
        ----------
        path: Needs 
                1. Crime dataset path
                2. Crime type path
                3. Police District dataset
                4. Ouptut file path
        crime_code  : Crime Nodes unique code
        police_code : Police station Nodes unique code
        Returns:
        -------
        Nothing
        """

        #Get the path for the datasets
        self.crime_path = path[0]
        self.type_path = path[1]
        self.police_path = path[2]
        self.output_path = path[3]
        self.police_code = police_code
        self.crime_code = crime_code

        self._build_police_code ()
        self._crime_police_network ()

    def get_network (self):
        """Returns a dict containing edge list from community to crime type. """

        return (self.police_crime)
    
    def _build_crime_type (self):
        """Build a dictionary of crime types according to chicago police website. """

        #Declare a dictionary of crime type
        self.crime_type = {}

        #Read csv file
        with open(self.type_path, 'rt') as f:

            i = 0
            j = 0

            reader = csv.reader(f)

            for row in reader:

                #Skip first line
                if (i == 0):
                    i += 1
                    continue

                temp = row[0]

                if (temp[0] == '0'):
                    self.crime_type[temp[1:]] = self.crime_code + j
                else:
                    self.crime_type[temp] = self.crime_code + j
                j += 1

    def _build_police_code (self):
        """ Build police station dictionary. """

        self.police_dict = {}
        df = pd.read_csv(self.police_path, usecols=['DISTRICT'])

        for i, district in enumerate (df.DISTRICT):
            try: 
                self.police_dict[int (district)] = self.police_code + i
            except ValueError:
                self.police_dict[0] = self.police_code + i

    def _crime_police_network (self):
        """ Build crime police network. """

        self._build_crime_type ()
        self.police_crime = {}
        df = pd.read_csv(self.crime_path, usecols=['IUCR','District'])

        for i, district in enumerate (df.District):
            try:
                district = int (district)
            except ValueError:
                continue

            if district in self.police_dict:
                try:
                    district = self.police_dict[district]
                except KeyError:
                    continue
                if (df.ix[i, 'IUCR'] in self.crime_type):
                    crime = self.crime_type[df.ix[i, 'IUCR']]
                else:
                    continue

                if district not in self.police_crime:
                    self.police_crime[district] = {}
                    self.police_crime[district][crime] = 1
                else:
                    try: 
                        self.police_crime[district][crime] += 1
                    except KeyError:
                        self.police_crime[district][crime] = 1
    
#path = []
#path.append ("../../Data/2015/crime_2015.csv")
#path.append ("../../Data/Static/IUCR.csv")
#path.append ("../../Data/Static/Map_police_community.csv")
#path.append ("../Data/2015/network.xml")
p = Path ()
if (__name__ == '__main__'):
    for year in range (2011, 2016):
        print ("For year: {}".format (year))
        a = CrimePoliceNetwork (p.get_path (year, "police_crime"))
        if (year == 2015):
            print (a.get_network ())
