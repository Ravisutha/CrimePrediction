#!/usr/bin/python3

#Author :   Ravisutha Sakrepatna Srinivasamurthy
#Project:   Analysis of Chicago Crime Data

import pandas as pd
from pandas import DataFrame, read_csv
import csv
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class community_map:
    """Maps location data (in the required dataset) to community dataset."""

    def __init__(self, path):

        #Get the path for the dataset containing location and polygon shape
        self.community_path = path[0]
        self.dataset_path = path[1]
        self.input_path = path[2]
        self.output_path = path[3]

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

        #Remove bracketsfrom shapely.geometry import Point
        point = point.replace ("(", "")
        point = point.replace (")", "")

        #Parse the location string
        loc = point.split (',')

        #Returns tuple
        return (float (loc[0]), float (loc[1]))

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

    def get_location (self):
        """Get the location in the dataset."""

        self.school_id = {}

        with open(self.dataset_path, 'rt') as f:
            reader = csv.reader(f)

            #Read latitude and longitude for each row in the dataset
            for i, row in enumerate (reader):

                #Skip first line
                if (i > 0):
                    #For school dataset, get location
                    location = Point (self._string_location (row[-1]))

                    #Map this location to community
                    self.school_id[int (row[0])] = self._map (location)

        return (self.school_id)

    def _map (self, location):
        """Map location data to community."""

        for community in self.community_dict:
            #If the location belongs to this community
            if (self.community_dict[community].contains(location) == True):
                return (community)

        #If it doesn't belong to any community, return garbage
        return (0)

    def map_location (self, year=2015, month=None):
        """ Map school location to community. """
        
        community = [] 
        bad_id = 0

        df = self._filter (year, month)

        for i, row in enumerate (df.values):
            """ Iterate over the School ID column and find its corresponding community. """

            sh_id = int (row[1])
            if (sh_id in self.school_id):
                community.append (self.school_id[sh_id])

            else:
                community.append (0)
                bad_id += 1

        print ("Number of bad ids = {}".format (bad_id))

        df['Community'] = pd.Series  (community, index=df.index)

        df.to_csv (self.output_path)

    def _filter (self, year=2015, month=None):
        """ Filter data based on years. """

        df = pd.read_csv (self.input_path, skiprows=[0])

        df_year = df.query ("Year==" + str(year))

        if (month != None):
            df_year = df.query ("Month==" + str (month))

        return (df_year)


for year in range (2011, 2016):
    for month in range (1, 13):
        path = []
        path.append ("../../Data/Static/community.csv")
        path.append ("../../Data/Static/chicago_public_schools.csv")
        path.append ("../../Data/Static/average_act.csv")
        path.append ("../../Data/" + str (year) + "/" + str (month) + "/map_average_act_" + str (year)+ "_" + str (month) + ".csv")

        a = community_map (path)
        a.get_location ()
        a.map_location (year)
