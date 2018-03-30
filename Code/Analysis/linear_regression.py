#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:07:35 2017

@author: rsakrep
"""

import pickle
import pandas as pd
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import normalize
from scipy.interpolate import *
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AR
from similarity import FindSimilarity
import sys
from path import Path
from crime_type_map import map_codes

class Regression:
    """ Use Regression to predict next crime pattern. """
    
    def __init__ (self):
        
        self.once = 0
        self.map_crime_types ()
    
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
            for month in range (1, 13):
            #for month in range (1):
                #Get similarity matrix for the years 2011, 2012, 2013, 2014
                [arr, attr] = self.get_sim_matrix (year, month)
                
                #Stack similarity matrix for all months for a given year
                sim_arr[year][month] = arr
                attr_arr[year][month] = attr

        #Loop over all year and months. Predict for 2015
        for month in range (1, 13):
        #for month in range (1):
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

                self.result[month].append ([float (t_output), float (out)])
                self.error[month].append((abs(t_output - out) / t_output))

        return (self.result)

    def map_crime_types (self):
        """
        Add all the weights related to the given primary crime type and return.
        """

        self.map_crime = map_codes ("../../Data/Static/IUCR.csv", 10000)

    def _get_data (self):
        """ Get data 
        Input Parameters:
            None
        Returns:
            Tuple of similarity matirx and attributes
        """

        #Initialize the lists
        self.sim_arr = {}
        self.attr_arr = {}

        #Loop through years 2011-2015 and get similar communities
        for year in range (2011, 2016):
            self.sim_arr[year] = {}
            self.attr_arr[year] = {}

            for month in range (1, 13):
                #Get similarity matrix for the years 2011, 2012, 2013, 2014
                [arr, attr] = self.get_sim_matrix (year, month)

                #Stack similarity matrix for all months for a given year
                self.sim_arr[year][month] = arr
                self.attr_arr[year][month] = attr

        return (self.sim_arr, self.attr_arr)

    def _auto_regression_input (self):
        """ Find the optimal lag using auto-correlation. 
        Prameter:
            year: Year for which auto-correlation is required.
        Output :
            Returns a dictionary of list containing optimal lag for each month
        """

        auto_corr_train = {}
        auto_corr_test = {}

        # Build just output data
        for comm in range (1, 78):
            auto_corr_train[comm] = {}
            auto_corr_test[comm] = {}
            for month in range (1, 13):
                auto_corr_train[comm][month] = []
                auto_corr_test[comm][month] = []

                # Loop through all years and collect crime data
                for year in range (2011, 2015):
                    auto_corr_train[comm][month].append (self.add_weights (self.attr_arr[year][month]["crime"][comm], ["HOMICIDE"]))

                year = 2015
                auto_corr_test[comm][month].append (self.add_weights (self.attr_arr[year][month]["crime"][comm], "HOMICIDE"))

        return (auto_corr_train, auto_corr_test)

    def auto_regression (self):
        """ Returns auto regression output. """

        #Output list
        self.result = {}
        self.error = {}

        # Get data
        self._get_data ()

        # Find the optimal lag
        (train, test) = self._auto_regression_input ()
        print ("Train:")
        print (train)
        print ("Test")
        print (test)

        # Train
        for month in range (1, 13):
            self.result[month] = []
            self.error[month] = []

            for community in range (1, 78):
                train1 = np.array (train[community][month])
                train1 = train1.reshape (-1, 1)
                test1 = test[community][month]

                # Autoregression
                model = AR(train1)
                max_lag = 1
                model_fit = model.fit(max_lag)
                #print('Lag: %s' % model_fit.k_ar)
                #print('Coefficients: %s' % model_fit.params)

                # Test
                out = model_fit.predict(start=len(train1), end=len(train1)+len(test1)-1, dynamic=False)
                t_output = test1

                #print('predicted={}, expected={}'.format(predictions[community][month], test1))
                self.result[month].append ([float (t_output[0]), float (out[0])])
                self.error[month].append((abs(t_output - out) / t_output))

    def regression_svr (self, sim_num=3):
        """ Using libsvm in scklearn, preform regression. """

        #Output list
        self.result = {}
        self.error = {}

        # Get data
        self._get_data ()

        #Loop over all year and months. Predict for 2015
        for month in range (1, 13):
            self.result[month] = []
            self.error[month] = []

            for comm_no in range (1, 78):
                matrix = []
                output = []

                for year in range (2011, 2015):
                    arr = self.sim_arr[year][month]
                    attr = self.attr_arr[year][month]

                    #Get top two similar communities for this community
                    index = self.n_similar_communities (sim_num, comm_no, arr)

                    [temp_matrix, temp_output] = self.process_attributes (index, attr)
                    matrix.append (temp_matrix)
                    output.append (temp_output)

                #Get the attributes for 2015
                index = self.n_similar_communities (sim_num, comm_no, self.sim_arr[2015][month])
                [test, t_output] = self.process_attributes (index, self.attr_arr[2015][month])

                #Convet to np array
                matrix = np.array (matrix)
                output = np.array (output)

                #Polynomial regression with degree 2
                clf = svm.SVR ()
                sc = MinMaxScaler(feature_range=(0, 1))
                X_ = sc.fit_transform (matrix)

                # Train
                clf.fit (X_, output)

                # Test
                test = np.array (test, ndmin=2)
                t_output = np.array (t_output, ndmin=2)
                predict_ = sc.transform (test)

                out = clf.predict (predict_)
                print ("\t(Actual, Predicted) = ({}, {})".format (t_output, out))

                self.result[month].append ([float (t_output), float (out)])
                self.error[month].append((abs(t_output - out) / t_output))

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
        """ Convert attributes as inputs to linear regression.
        Input parametes:
            index: Community list for which parameters are needed
            attr: Attribute dictionary
        Returns:
            List of attributes
        """

        mat = []
        output = []
        for itr, i in enumerate (index):
            comm = i

            #Crime Types (Total crimes)
            if (itr == 0):
                crime = self.add_weights (attr["crime"][comm], ["HOMICIDE", "ASSAULT", "BURGLARY"])
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

    def add_weights (self, in_dict, crime_types=["Full"]):
        """ Adds weight for the given dictionary. """

        weights = 0
        
        if (crime_types[0] == "Full"):
            for code in in_dict:
                try:
                    weights += float (in_dict[code])
                except ValueError:
                    #print ("Unknown value:", in_dict[code])
                    continue
        else:
            for crime_type in crime_types:
                for code in self.map_crime[crime_type]:
                    try:
                        weights += float (in_dict[code])
                    except ValueError:
                        continue
                    except KeyError:
                        continue

        return weights

    def print_results (self, path):
        """ Print the output to a file. """

        np.set_printoptions(suppress=True)

        print (self.result)
        for month in self.result:
            #print ("Month: %s"%month)
            #result = np.array(self.result[month])
            #result = self.result[month]
            #print (result.shape)
            #print (self.result[month])
            #print("Result: ", sum(self.error[month])) 
            #np.squeeze (result)
            #print (result.shape)

            np.savetxt (path[month], self.result[month])

    def plot_results (self, path, month=1):
        """ Plots the graph of actual and predicted crimes for given month"""

        result = self.result[month]

        communities = []
        for i in range (77):
            communities.append (i)

        actual = []
        predict = []
        for out in result:
            actual.append (out[0])
            predict.append (out[1])

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
        plt.savefig (path)
    plt.show ()

def main ():
    """ Program starts executing. """

    path = {}
    for month in range (1, 13):
        path[month] = "../../Data/Total_Data/Output/Poly/predict_" + str(month) + ".csv"

    reg = Regression ()
    reg.linear_regression ()
    #reg.auto_regression ()
    #reg.regression_svr ()
    reg.print_results (path)

    #for i in range (1, 13):
    #for i in range (1):
    #    print("Month: {}".format(i))
        #reg.plot_results ("../../Data/Total_Data/Output/plot_{}.png".format(i), i)
        #reg.plot_results ("../../Data/Total_Data/Output/plot_{}.png".format(i), i)

main ()
