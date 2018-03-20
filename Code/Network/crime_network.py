#!/usr/bin/python3

#Author :   Ravisutha Sakrepatna Srinivasamurthy
#Project:   Analysis of Chicago Crime Data

import datetime
import csv
import numpy as np

class Crime_Network:
    """Built to analyse Chicago Crime Data. \\
    Given a dataset with community details, a network is built."""

    def __init__(self, path, offset=10000):
        """ Accepts the path of crime type data, community data and IUCR data. """

        #Get the path for the required three data
        self.crime_path = path[0];
        self.type_path = path[1]
        self.comm_path = path[2];

        #Get node id offset
        self.offset = offset

        #Build the crime type dictionary
        self._built_crime_type ()

        #Build crime network and return tuples
        self._crime_network ()

    def get_network (self):
        """Returns a dict containing edge list from community to crime type. """

        return (self.community)

    def _built_crime_type (self):
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
                    self.crime_type[temp[1:]] = self.offset + j
                else:
                    self.crime_type[temp] = self.offset + j
                j += 1

    def _convert_date (self, t):
        """ Convert date and time into usable format. """

        #Get date, time and am/pm 
        [date, time, m] = t.split (" ")

        #Extract month, date and year from date string
        [month, date, year] = date.split ("/")

        #Covert string to integer
        month = int (month)
        date = int (date)
        year = int (year)

        #Calculate weekday: (0: Sunday, 6: Saturday)
        d = datetime.date (year, month, date)
        weekday = d.weekday()

        [t, temp, temp] = time.split (":")
        t = int (t)

        #Classify time into four categories: Morning, Afternoon, Evening, Earlymorning
        if (m == "PM"):
            if (t >= 1 and t < 6):                  #1PM - 5PM
                time = 2
            elif (t >= 6 and t <= 11):              #6PM - 11PM
                time = 3
            else:                                   #12PM
                time = 1
        if (m == "AM"):
            if (t < 7):                             #1AM - 6AM
                time = 4            
            elif (t >= 7 and t <= 11):              #7AM - 11:59AM
                time = 1
            else:                                   #12AM
                time = 4

        return (time, weekday)

    #Crime network
    def _crime_network (self):
        """Connect crimes and community. """

        self.community = {}

        #Initialize the community dictionary
        for i in range (1, 78):
            self.community[i] = {}

        #Open Crimes dataset
        with open (self.crime_path, "rt") as f:

            reader = csv.reader (f)
            i = 0

            for row in reader:

                #Skip first line
                if (i == 0):
                    i += 1
                    continue

                #Get date information
                date = row[3]
                (t, w) = self._convert_date (date)

                #Crime Type
                temp = row[5]

                if (temp[0] == '0'):
                    if (not(temp[1:] in self.crime_type)):
                        #print ("Bad Data: Crime Type {}".format (temp))
                        continue
                    else:
                        c_type = self.crime_type[temp[1:]]
                else:
                    if (not(temp in self.crime_type)):
                        #print ("Bad Data: Crime Type {}".format (temp))
                        continue
                    else:
                        c_type = self.crime_type[temp]

                #Community Type
                try:
                    comm = int (float (row[14]))
                except ValueError:
                    continue

                # Some garbage community
                if (not (comm in self.community)):
                    continue

                # Create new crime type if not present (Dictionary of
                # dictionary)
                if (not (c_type in self.community[comm])):
                    self.community[comm][c_type] = 1
                else:
                    self.community[comm][c_type] += 1

                i += 1

def main ():
    """ Main function. """
    path = []
    path.append ("../../Data/2015/crime_2015.csv")
    path.append ("../../Data/Static/IUCR.csv")
    path.append ("../../Data/Static/community.csv")

    a = Crime_Network (path)
    a.get_max ()

if (__name__=='__main__'):
    main ()
