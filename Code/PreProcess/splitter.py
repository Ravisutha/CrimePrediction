#!/usr/bin/python3
#Aim    :   Splits the overall data into years and months
#Author :   Ravisutha Sakrepatna Srinivasamurthy
#Project:   Analysis of Chicago Crime Data

import os
import pandas as pd

class Splitter:
    """Splits given data into seperate files"""

    def __init__(self, data_path, debug=False):
        """ Get data_path and output path. """

        #Get the path for the dataset and list of output files
        self.dataset_path = data_path
        self.debug = debug
        self._init_df ()

    def _print (self, arg):#Split the data into years
        """ Print only if debug mode is ON. """
        
        if (self.debug != False):
            print (arg)

    def _init_df (self):
        """ Initialize dataframe. """
        
        df = pd.read_csv (self.dataset_path)
        
        #df['date'] = pd.to_datetime (df['Date'])
        #df['date'] = pd.to_datetime (df['CREATION DATE'])
        #df['date'] = pd.to_datetime (df['Creation Date'])
        df['date'] = pd.to_datetime (df['DATE SERVICE REQUEST WAS RECEIVED'])
        
        self.df = df
        
        print (self.df.head ())
            
    def data_split_year (self, output_path, year):
        """Splits data according to years. 
            Inputs: Ouput path and year."""
        
        df = self.df
        
        df = df[df.date.dt.year == year]
        
        print ("For year {}:".format (year))
        print (df.head ())
        print (df.tail ())#Split the data into years
        
        df.to_csv (output_path)
        
    def data_split_months (self, output_path, year, month):
        """Splits data according to months. 
        Input: Ouput path and year."""
        
        df = self.df
        
        df = df[(df.date.dt.year == year) & (df.date.dt.month == month)]
        
        print ("For year {}:".format (year))
        print (df.head ())
        print (df.tail ())
        
        df.to_csv (output_path, index=False)

        
def main ():
    """ Execution starts. """
    
    #if (__name__ == '_main__'):
        
    #data_path = "../../Data/Total_Data/total_crime.csv"
    #data_path = "../../Data/Total_Data/sanitation_community.csv"
    #data_path = "../../Data/Total_Data/vehicles.csv"
    #data_path = "../../Data/Total_Data/pot_holes.csv"
    #data_path = "../../Data/Total_Data/trees.csv"
    data_path = "../../Data/Total_Data/vacant.csv"
        
    split = Splitter (data_path)
        
    for year in range (2001, 2016):
        #Create path string
        directory = "../../Data/" + str(year) + "/"
        #file = "sanity_" + str (year) + ".csv"
        #file = "vehicles_" + str (year) + ".csv"
        #file = "lights_one_" + str (year) + ".csv"
        #file = "pot_holes_" + str (year) + ".csv"
        #file = "lights_alley_" + str (year) + ".csv"
        #file = "trees_" + str (year) + ".csv"
        file = "vacant_" + str (year) + ".csv"
        
        #Try making a directory
        try:
            os.mkdir (directory)
        except FileExistsError:
            pass
        
        output_path =  directory + file
        
        #Split the data into years
        split.data_split_year (output_path, year)
        
        for month in range (1, 13):
            sub_directory = directory + str(month) + "/"
            
            #Try making a directory
            try:
                os.mkdir (sub_directory)
            except FileExistsError:
                pass
            
            #file = "sanity_" + str (year) + "_" + str (month) + ".csv"
            #file = "vehicles_" + str (year) + "_" + str (month) + ".csv"
            #file = "lights_one_" + str (year) + "_" + str (month) + ".csv"
            #file = "pot_holes_" + str (year) + "_" + str (month) + ".csv"
            #file = "lights_alley_" + str (year) + "_" + str (month) + ".csv"
            #file = "trees_" + str (year) + "_" + str (month) + ".csv"
            file = "vacant_" + str (year) + "_" + str (month) + ".csv"
            output_path =  sub_directory + file
            
            #Split the data into years
            split.data_split_months (output_path, year, month)
main ()
