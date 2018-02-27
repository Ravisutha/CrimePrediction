#!/usr/bin/python3

#Author :   Ravisutha Sakrepatna Srinivasamurthy
#           Saroj Kumar Dash
#Project:   Analysis of Chicago Crime Data

import networkx as nx
import csv
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from path import Path 

class community_map:
    """Maps location data (in the required dataset) to community dataset."""

    def __init__(self, path):

        #Get the path for the dataset containing location and polygon shape
        self.community_path = path[0]
        self.dataset_path = path[1]
        self.output_path = path[2]

        #Build the community location dictionary
        self.build_comm_dict ()

    def _preprocess_string (self, poly):
        """Covert string type to polygon type. """

        polygon = []
        poly = poly[16:-3].split(',')

        for point in poly:
            temp = point.split (' ')
            if (len (temp) > 2):
                temp[1] = temp[1].replace ('(', '')
                temp[2] = temp[2].replace (')', '')

                latitude = float (temp[1])
                longitude = float (temp[2])

            else:
                temp[0] = temp[0].replace ('(', '')
                temp[1] = temp[1].replace (')', '')

                if (temp[0][0] == '('):
                    temp[0] = temp[0][1:]
                if (temp[1][-1] == ')'):
                    temp[1] = temp[1][:-2]

                latitude = float(temp[0])
                longitude = float(temp[1])

            polygon.append ((latitude, longitude))

        return (Polygon(polygon))

    def _string_location (self, point):
        """Accepts strings and returns location as tuple. """

        #Remove brackets
        point = point.replace ("(", "")
        point = point.replace (")", "")

        #Parse the location string
        loc = point.split (',')

        #Returns tuple
        try:
            return (float (loc[1]), float (loc[0]))
        except IndexError:
            return ((0, 0))

    def build_comm_dict (self):
        """Build a dictionary of communitites gis boundary."""

        with open(self.community_path, 'rt') as f:
            reader = csv.reader(f)

            self.community_dict = {}
            i = 0

            #Store the polygon data in dictionary
            for row in reader:
                if (i != 0):
                    self.community_dict[int(row[-3])] = self._preprocess_string (row[0])

                i += 1

    def _get_location (self):
        """Get the location in the dataset."""

        locations = []
        i = 0

        with open(self.dataset_path, 'rt') as f:
            reader = csv.reader(f)

            #Read latitude and longitude for each row in the dataset
            for row in reader:
                if (i != 0):
                    #For police station data
                    #locations.append (Point(float(row[-2]), float(row[-3])))

                    #For library dataset
                    #locations.append (Point (self._string_location (row[-1])))

                    #For vacant dataset
                    locations.append (Point (self._string_location (row[-2])))

                i += 1

        return (locations)

    def map_location (self):
        """Map location data to community."""

        k = 0

        locations = self._get_location ()

        flag = 0
        for location in locations:
            for community in self.community_dict:
                if (self.community_dict[community].contains(location) == True):
                    locations[k] = community
                    flag = 1
                    break
            #Check if the location could be mapped
            if (flag == 0):
                #Assign garbage if location couldn't be mapped to any community
                locations[k] = 0
            else:
                flag = 0
            k += 1
        self._write_csv (locations)

    def _write_csv (self, locations):
       """ Write a new csv file. """

       with open(self.dataset_path,'r') as csvinput:
           with open(self.output_path, 'w') as csvoutput:
               writer = csv.writer(csvoutput, lineterminator='\n')
               reader = csv.reader(csvinput)
               all = []
               row = next(reader)
               row.append('Community')
               all.append(row)

               k = 0
               for row in reader:
                   row.append(locations[k])
                   all.append(row)
                   k += 1

               writer.writerows(all)

for year in range (2011, 2016):
    #Only for lib:
    #path.append ("../Data/Static/libraries.csv")
    
    path = []
    for month in range (1, 13):
        path.append ("../../Data/Static/community.csv")
        path.append ("../../Data/" + str(year) + "/" + str (month) + "/vacant_" + str(year) + "_" + str (month) + ".csv")
        path.append ("../../Data/" + str(year) + "/" + str (month) + "/map_vacant_" + str(year) + "_" + str (month) + ".csv")

        print (path)
        a = community_map (path)
        a.map_location ()
