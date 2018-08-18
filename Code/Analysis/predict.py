#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:07:35 2017

@author: Ravisutha Sakrepatna Srinivasamurthy 
"""

import os
import pickle
import pandas as pd
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
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
import pickle
import time

class Regression:
    """ Use Regression to predict next crime pattern. """
    
    def __init__ (self, init_path, methods=["Poly"], crime_types=["HOMICIDE"], save=True, plot=True):
        
        self.once = 0

        # Get crime types
        self.map_crime_types ()

        # Get data
        self._get_data (pick=False)

        # Decide on what to iterate
        if crime_types != -1:
            itr = crime_types
        else:
            itr = self.map_crime

        # For each crime type, predict the number of crimes
        sim_num = 1
        for method in methods:
            for crime_type in self.map_crime: 
                if (method == "Poly"):
                    self.linear_regression([crime_type],sim_num)
                elif (method == "Auto"):
                    self.auto_regression ([crime_type])
                else:
                    self.regression_svr ([crime_type],sim_num)

                if save:
                    self.print_results (init_path, method, crime_type)

                if plot:
                    self.plot_results (init_path, method, crime_type)
    


    def dimensionality_reduction (self, train_matrix, test, dim):
        """
        Reduces the dimension of the input feature matrix.
        -----------------------
        Input:
        in_matrix: input feature matrix
        num: Number of features to be kept
        ----------------------
        Returns:
        reduced feature matrix
        """
        
        pca = PCA (dim)
        out_matrix = pca.fit_transform (train_matrix)
        test = pca.transform (test)
        
        return [out_matrix, test]
        
    def map_crime_types (self):
        """
        Add all the weights related to the given primary crime type and return.
        """

        self.map_crime = map_codes ("../../Data/Static/IUCR.csv", 10000)

    def _get_data (self, pick=True):
        """ Get data 
        Input Parameters:
            None
        Returns:
            Tuple of similarity matirx and attributes
        """

        if (pick == False):
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

            pickle.dump (self.sim_arr, open( "sim_arr.p", "wb" ))
            pickle.dump (self.attr_arr, open( "attr_arr.p", "wb" ))

        else:
            self.sim_arr = pickle.load (open( "sim_arr.p", "rb" ))
            self.attr_arr = pickle.load (open( "attr_arr.p", "rb" ))

        return (self.sim_arr, self.attr_arr)

    def linear_regression (self, crime_type,sim_num = 1):
        """ Perform linear regression on the given data. (\alpha1 * sim1 + \alpha2 * sim2 = totat_crime)"""
        
        #Output list
        self.result = {}
        self.error = {}

        #Initialize the lists
        sim_arr = self.sim_arr
        attr_arr = self.attr_arr

        #Number of similar community data
        #sim_num = 1

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
                    #print(str(index) + " : " + str(comm_no))
                    #index = [comm_no-1]

                    #[temp_matrix, temp_output] = self.process_attributes (index, attr, month=month, extra=attr_arr[year], crime_type=crime_type)
                    [temp_matrix, temp_output] = self.process_attributes (index, attr, month=-1, extra=attr_arr[year], crime_type=crime_type)
                    matrix.append (temp_matrix)
                    output.append (temp_output)

                #Get the attributes for 2015
                index = self.n_similar_communities (sim_num, comm_no, sim_arr[2015][month])
                [test, t_output] = self.process_attributes (index, attr_arr[2015][month], month=-1, extra=attr_arr[2015], crime_type=crime_type)

                #Convet to np array
                matrix = np.array (matrix)
                output = np.array (output)
                test = np.array (test, ndmin=2)
                t_output = np.array (t_output, ndmin=2)

                #Normalize
                sc1 = MinMaxScaler(feature_range=(0, 1))
                sc2 = MinMaxScaler(feature_range=(0, 1))
                matrix = sc1.fit_transform (matrix)
                test = sc1.transform (test)
                output = sc2.fit_transform (output)

                print(matrix)
                print(output)
                input()
                #Polynomial regression with degree 2
                poly = PolynomialFeatures(degree=2)
                X_ = poly.fit_transform (matrix)

                # Train
                clf = LinearRegression ()
                clf.fit (X_, output)

                # Predict
                predict_ = poly.fit_transform (test)
                out = clf.predict (predict_)
                out = sc2.inverse_transform (out)

                self.result[month].append ([float (t_output), float (out)])
                self.error[month].append((abs(t_output - out) / t_output))

        return (self.result)

        #Saroj Dash: Function that would generate the baseline result.inpsired from auto regression function
        def linear_regression2 (self, crime_type,sim_num = 1):
        """ Returns auto regression output. """

        #Output list
        self.result = {}
        self.error = {}

        # Find the optimal lag
        (train, test) = self._auto_regression_input (crime_type)

        
        #input()

        # Train
        for month in range (1, 13):
            self.result[month] = []
            self.error[month] = []

            for community in range (1, 78):
                train1 = np.array (train[community][month])
                train1 = train1.reshape (-1, 1)
                test1 = test[community][month]

                print(train1)
                #input()
                # Autoregression
                # model = AR(train1)
                # max_lag = 1
                # model_fit = model.fit(max_lag)

                poly = PolynomialFeatures(degree=2)
                X_ = poly.fit_transform (train1)

                # Train
                clf = LinearRegression ()
                clf.fit (X_, output)

                # Test
                try:
                    out = model_fit.predict(start=len(train1), end=len(train1)+len(test1)-1, dynamic=False)
                except ValueError:
                    print ("Something went wrong")
                    out = [train1[-1]]
                    continue

                t_output = test1

                self.result[month].append ([float (t_output[0]), float (out[0])])
                self.error[month].append((abs(t_output - out) / t_output))

    def _auto_regression_input (self, crime_type):
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
                    auto_corr_train[comm][month].append (self.add_weights (self.attr_arr[year][month]["crime"][comm], crime_type))

                year = 2015
                auto_corr_test[comm][month].append (self.add_weights (self.attr_arr[year][month]["crime"][comm], crime_type))

        return (auto_corr_train, auto_corr_test)

    def auto_regression (self, crime_type):
        """ Returns auto regression output. """

        #Output list
        self.result = {}
        self.error = {}

        # Find the optimal lag
        (train, test) = self._auto_regression_input (crime_type)

        
        #input()

        # Train
        for month in range (1, 13):
            self.result[month] = []
            self.error[month] = []

            for community in range (1, 78):
                train1 = np.array (train[community][month])
                train1 = train1.reshape (-1, 1)
                test1 = test[community][month]

                print("Autoregression")
                print(train1)
                input()
                # Autoregression
                model = AR(train1)
                max_lag = 1
                model_fit = model.fit(max_lag)

                # Test
                try:
                    out = model_fit.predict(start=len(train1), end=len(train1)+len(test1)-1, dynamic=False)
                except ValueError:
                    print ("Something went wrong")
                    out = [train1[-1]]
                    continue

                t_output = test1

                self.result[month].append ([float (t_output[0]), float (out[0])])
                self.error[month].append((abs(t_output - out) / t_output))

    def regression_svr (self, crime_type, sim_num=1):
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

                    #[temp_matrix, temp_output] = self.process_attributes (index, attr, month=month, extra=self.attr_arr[year], crime_type=crime_type)
                    [temp_matrix, temp_output] = self.process_attributes (index, attr, month=-1, extra=self.attr_arr[year], crime_type=crime_type)
                    matrix.append (temp_matrix)
                    output.append (temp_output)

                #Get the attributes for 2015
                index = self.n_similar_communities (sim_num, comm_no, self.sim_arr[2015][month])
                #[test, t_output] = self.process_attributes (index, self.attr_arr[2015][month], month=month, extra=self.attr_arr[2015], crime_type=crime_type)
                [test, t_output] = self.process_attributes (index, self.attr_arr[2015][month], month=-1, extra=self.attr_arr[2015], crime_type=crime_type)

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

        #print("n value in n_similar_communities " + str(n))
        #n = 0
        #Sort and return n similar communities
        
        index_no = comm_no - 1
        req_sim = arr[index_no, :]

        sorted_arr = np.sort (req_sim)[::-1]
        index = np.argsort (req_sim)[::-1]

        for i, idex in enumerate (index):
            index[i] = idex + 1

        return (index[0:n])
        #return (index[index_no])

    def process_attributes (self, index, attr, month=-1, extra=False, crime_type=["HOMICIDE"]):
        """ Convert attributes as inputs to linear regression.
        Input parametes:
            index: Community list for which parameters are needed
            attr: Attribute dictionary
        Returns:
            List of attributes
        """

        mat = []
        output = []
        #print(len(index))

        for itr, i in enumerate (index):
            comm = i

            #Crime Types (Total crimes)
            if (itr == 0):
                crime = self.add_weights (attr["crime"][comm], crime_type)
                output.append (crime)

                if (extra != False):
                    for prev in range (1, month):
                        mat.append (self.add_weights (extra[prev]["crime"][comm], crime_type))

            
            # Number of Police Stations
            try:
                police = len (attr["police"][comm])
            except KeyError:
                police = 0
            mat.append (police)

            # #Number of visitors

            # #Sanity
            # try:
            #     sanity = attr["sanity"][comm][40000]
            # except KeyError:
            #     sanity = 0
            # mat.append (sanity)

            # #Vehicles
            # try:
            #     vehicles = attr["vehicles"][comm][50000]
            # except KeyError:
            #     vehicles = 0
            # mat.append (vehicles)

            # #Pot holes
            # try:
            #     pot_holes = attr["pot_holes"][comm][60000]
            # except KeyError:
            #     pot_holes = 0
            # mat.append (pot_holes)

            # #Lights one
            # try:
            #     light_1 = attr["lights_one"][comm][70000]
            # except KeyError:
            #     light_1 = 0
            # mat.append (light_1)

            # #Lights all
            # try:
            #     light_2 = attr["lights_all"][comm][80000]
            # except KeyError:
            #     light_2 = 0
            # mat.append (light_2)

            # #Lights alley
            # try:
            #     light_alley = attr["lights_alley"][comm][90000]
            # except KeyError:
            #     light_alley = 0
            # mat.append (light_alley)

            # #Trees
            # try:
            #     trees = attr["trees"][comm][100000]
            # except KeyError:
            #     trees = 0
            # mat.append (trees)

            # #Vacant buildings
            # try:
            #     vacant = attr["vacant"][comm][110000]
            # except KeyError:
            #     vacant = 0
            # mat.append (vacant)

            # #School
            # try:
            #     school = len (attr["school"][comm])
            # except KeyError:
            #     school = 0
            # mat.append (school)
            #print(mat)
            #print(output)

        
        return ([mat, output])

    def add_weights (self, in_dict, crime_types=["Full"]):
        """ Adds weight for the given dictionary. """

        weights = 0
        
        if (crime_types[0] == "FULL"):
            for code in in_dict:
                try:
                    weights += float (in_dict[code])
                except ValueError:
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

    def print_results (self, init_path, method, crime_type):
        """ Print the output to a file. """

        np.set_printoptions(suppress=True)


        for month in self.result:
            path = init_path + method + "/"+ crime_type
            os.makedirs (path, exist_ok=True)
            path = path + "/prediction_" + str (month) + ".csv"
            #print ("Writing results into", path)

            np.savetxt (path, self.result[month])
        
        

def main ():
    """ Program starts executing.
    Specify which prediction model which you want to run
    Recomended :
         method = ["Poly", "Auto", "SVR"]
         init_path = "../../Data/Total_Data/Output/"
         reg = Regression (methods=method, init_path=init_path, crime_types=-1, save=True, plot=False)
    """

    #method = ["Poly", "Auto", "SVR"]
    #method = ["Auto"]
    method = ["Poly"]

    #init_path = "../../Data/Total_Data/Output/Only-Crime/"
    #init_path = "/home/saroj/Documents/Data/Full-Crime/"
    init_path = "/media/saroj/Work/Clemson Semesters/Dr.SafroResearch/Backup MY Code/MyRepositoyr/Chicago_Crime_Project - Copy/Data/Only-Crime/"
    reg = Regression (methods=method, init_path=init_path, crime_types=-1, save=True, plot=False)

main ()
