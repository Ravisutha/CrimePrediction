#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:07:35 2017

@author: rsakrep
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import normalize
from scipy.interpolate import *
from sklearn.linear_model import LinearRegression
from similarity import FindSimilarity
import sys
from path import Path

class Regression:
    """ Use Regression to predict next crime pattern. """
    
    def __init__ (self):
        
        self.once = 0
    
    def linear_regression (self):
        """ Perform linear regression on the given data. (\alpha1 * sim1 + \alpha2 * sim2 = totat_crime)"""
        
        #Output list
        self.result = {}
        self.error = {}

        #Initialize the lists
        sim_arr = {}
        attr_arr = {}

        #Number of similar community data
        sim_num = 5

        #Loop through years 2011-2015 and get similar communities
        for year in range (2011, 2016):
            sim_arr[year] = {}
            attr_arr[year] = {}
            #for month in range (1, 13):
            for month in range (1):
                #Get similarity matrix for the years 2011, 2012, 2013, 2014
                [arr, attr] = self.get_sim_matrix (year, month)
                
                #Stack similarity matrix for all months for a given year
                sim_arr[year][month] = arr
                attr_arr[year][month] = attr
        #Loop over all year and months. Predict for 2015
        #for month in range (1, 13):
        for month in range (1):
            self.result[month] = []
            self.error[month] = []

            for comm_no in range (1, 78):
                matrix = []
                output = []

                for year in range (2011, 2015):
                    arr = sim_arr[year][month]
                    attr = attr_arr[year][month]

                    #Get top two similar communities for this community
                    index = self.n_similar_communities (sim_num, comm_no, arr)

                    [temp_matrix, temp_output] = self.process_attributes (index, attr)
                    matrix.append (temp_matrix)
                    output.append (temp_output)

                #Get the attributes for 2015
                index = self.n_similar_communities (sim_num, comm_no, sim_arr[2015][month])
                [test, t_output] = self.process_attributes (index, attr_arr[2015][month])

                #Convet to np array
                matrix = np.array (matrix)
                output = np.array (output)

                #Polynomial regression with degree 2
                poly = PolynomialFeatures(degree=2)
                X_ = poly.fit_transform (matrix)

                clf = LinearRegression ()
                clf.fit (X_, output)

                test = np.array (test, ndmin=2)
                t_output = np.array (t_output, ndmin=2)
                predict_ = poly.fit_transform (test)

                out = clf.predict (predict_)
                #print ("\t(Actual, Predicted) = ({}, {})".format (t_output, clf.predict (predict_)))
                print ("\t(Actual, Predicted) = ({}, {})".format (t_output, out))

                self.result[month].append ([t_output, out])
                self.error[month].append((abs(t_output - out) / t_output))

        return (self.result)
    
    def get_sim_matrix (self, year, month=1):
        """ Return similarity matrix for a given year. """

        #Get similarity matrix
        sim = FindSimilarity(year=year, month=month)
        [arr, G] = sim.get_similarity ()
        attr = sim.get_attributes ()

        return ([arr, attr])

    def n_similar_communities (self, n, comm_no, arr):
        """ Returns top "n" similar community numbers. """

        #Sort and return n similar communities
        index_no = comm_no - 1
        req_sim = arr[index_no, :] 

        sorted_arr = np.sort (req_sim)[::-1]
        #print (sorted_arr)
        index = np.argsort (req_sim)[::-1]
        #print (index)
            
        for i, idex in enumerate (index):
            index[i] = idex + 1

        return (index[0:n])

    def process_attributes (self, index, attr):
        """ Convert attributes as inputs to linear regression. """
           
        mat = []
        output = []
        for itr, i in enumerate (index):
            comm = i
            #print ("Community: {}".format (comm))

            #Crime Types (Homicide)
            if (itr == 0):
                crime = self.add_weights (attr["crime"][comm])
                output.append (crime)

            #Number of Police Stations
            try:
                police = len (attr["police"][comm])
            except KeyError:
                police = 0
            mat.append (police)

            #Number of visitors

            #Sanity
            try:
                sanity = attr["sanity"][comm][40000]
            except KeyError:
                sanity = 0
            mat.append (sanity)

            #Vehicles
            try:
                vehicles = attr["vehicles"][comm][50000]
            except KeyError:
                vehicles = 0
            mat.append (vehicles)

            #Pot holes
            try:
                pot_holes = attr["pot_holes"][comm][60000]
            except KeyError:
                pot_holes = 0
            mat.append (pot_holes)

            #Lights one
            try:
                light_1 = attr["lights_one"][comm][70000]
            except KeyError:
                light_1 = 0
            mat.append (light_1)

            #Lights all
            try:
                light_2 = attr["lights_all"][comm][80000]
            except KeyError:
                light_2 = 0
            mat.append (light_2)

            #Lights alley
            try:
                light_alley = attr["lights_alley"][comm][90000]
            except KeyError:
                light_alley = 0
            mat.append (light_alley)

            #Trees
            try:
                trees = attr["trees"][comm][100000]
            except KeyError:
                trees = 0
            mat.append (trees)

            #Vacant buildings
            try:
                vacant = attr["vacant"][comm][110000]
            except KeyError:
                vacant = 0
            mat.append (vacant)

            #School
            try:
                school = len (attr["school"][comm])
            except KeyError:
                school = 0
            mat.append (school)

        return ([mat, output])

    def add_weights (self, in_dict):
        """ Adds weight for the given dictionary. """

        weights = 0 
        for code in in_dict:
            try:
                weights += float (in_dict[code])
            except ValueError:
                #print ("Unknown value:", in_dict[code])
                continue

        return weights
    
    def print_results (self, path):
        """ Print the output to a file. """

        np.set_printoptions(suppress=True)

        for month in self.result:
            result = np.array(self.result[month])
            print("Result: ", sum(self.error[month])) 
        
            np.savetxt (path[month], result)

    def plot_results (self, path, month=1):
        """ Plots the graph of actual and predicted crimes for given month"""

        result = self.result[month]

        communities = []
        for i in range (77):
            communities.append (i)

        actual = []
        predict = []
        for out in result:
            actual.append (out[0][0])
            predict.append (out[1][0])

        if (month == 1):
            title = "Number of crimes for the month January in the 77 communities"
        if (month == 2):
            title = "Number of crimes for the month February in the 77 communities"
        if (month == 3):
            title = "Number of crimes for the month March in the 77 communities"
        if (month == 4):
            title = "Number of crimes for the month April in the 77 communities"
        if (month == 5):
            title = "Number of crimes for the month May in the 77 communities"
        if (month == 6):
            title = "Number of crimes for the month June in the 77 communities"
        if (month == 7):
            title = "Number of crimes for the month July in the 77 communities"
        if (month == 8):
            title = "Number of crimes for the month August in the 77 communities"
        if (month == 9):
            title = "Number of crimes for the month September in the 77 communities"
        if (month == 10):
            title = "Number of crimes for the month October in the 77 communities"
        if (month == 11):
            title = "Number of crimes for the month November in the 77 communities"
        if (month == 12):
            title = "Number of crimes for the month December in the 77 communities"

        #print (title)
        print("Result: {}".format(sum(self.error[month])))
        plt.figure ()
        #print (communities)
        #print (actual)
        #print (predict)
        plt.plot (communities, actual, label="Actual number of crimes")
        plt.plot (communities, predict, label="Predicted number of crimes")
        #plt.title (title)
        plt.xlabel("Cummunities")
        plt.ylabel ("Number of crimes")
        plt.legend ()
        plt.show ()
        plt.savefig (path)
                
def main ():
    """ Program starts executing. """
    
    path = {}
    for month in range (1, 13):
        path[month] = "../../Data/Total_Data/Output/predict_" + str(month) + ".csv"

    reg = Regression ()
    reg.linear_regression ()

    #for i in range (1, 13):
    for i in range (1):
        print("Month: {}".format(i))
        reg.plot_results ("../../Data/Total_Data/Output/plot_{}.png".format(i), i)

main ()
