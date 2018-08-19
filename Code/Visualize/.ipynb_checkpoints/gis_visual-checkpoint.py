#!/usr/bin/python3
# coding: utf-8

# In[88]:
import geopandas as gpd
#import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
#get_ipython().magic(u'matplotlib inline')

class community_map:
    """makes an object of the map of the shape file"""
    
    def __init__(self,shpFilePath):
        """ @shpFilePath: Accept the filepath of the shapefile"""
        
        self._fp = shpFilePath
        self._geodf_Map = gpd.read_file(shpFilePath)
        #adding my own column name for storing similarity, default value=0
        self._geodf_Map['my_similarity'] = 0
      

    def showData(self,iColName=[]):
        """Shows the geoDataframe of the map.Takes column name in a list, by default it shows all the Columns"""
        
        if len(iColName) == 0:
            df = pd.DataFrame(self._geodf_Map)
            print(self._geodf_Map)
        else:
            df = pd.DataFrame(self._geodf_Map,columns=iColName)
            print(df)
        
        
    def plotMap(self,year,iColorMap="Reds",iAttr='my_similarity',bShow=True):
        """plots the map that is present in the object. @bShow: if True plots the map. if false the saves the map"""
        cwd = os.getcwd()
        myPath = cwd +"/../../Data/Visualize/"
        ax = self._geodf_Map.plot(column=iAttr,cmap=iColorMap,figsize=(30,10),
                                  linewidth=1,k=9,edgecolor='black')
        
        if True == bShow:
            plt.show()
        else:
            plt.savefig(myPath+iAttr+"_"+str(year))

    def fillSimilarityCol(self,inSimilarityFP,inComm):
        """ this function fills the column my_similarity which is created in the constructor,
            @inSimilarityFP: input parameter which takes the file path of the similarity csv file for the community.
            @inComm: Community number whose similarity map we want to find out
            return Value: returns the newcolumn Name"""
        
        simdf = pd.read_csv(inSimilarityFP,names = ["similarity", "src", "dst"])
        commdf = self._geodf_Map.copy() #to be on safe side we are making a copy
        #print(commdf.columns)

        arr = []   #to extract all dst,<similarity value> pairs 
        for i,row in enumerate(simdf['similarity']):
            if( inComm == simdf.iloc[i]['src']):
                dst = simdf.iloc[i]['dst']
                sim = simdf.iloc[i]['similarity']
                arr.append((dst,sim))

        #print(arr)

        mymax = 0
        for row in arr:
            for it,nb in enumerate(commdf.area_num_1):
                if(int(nb) == int(row[0])):
                    norm_similarity = (int)((row[1])*128)
                    commdf.my_similarity[it] = norm_similarity
                    if mymax < norm_similarity:
                        mymax = norm_similarity
        
        for ith,commNum in enumerate(commdf.area_num_1):
            if int(commNum) == int(inComm):
                commdf.my_similarity[ith] = mymax+1


        #print("mymax: " + str(mymax))
        #print(self._geodf_Map)
        newCol = "my_similarity_"+str(inComm)
        self._geodf_Map[newCol] = commdf['my_similarity']
        #print("NewColumn Values")
        #print(self._geodf_Map[newCol])

        return newCol
    
    
    
    def fillSimilarityCol1(self,inSimilarityFP,inComm):
        
        simdf = pd.read_csv(inSimilarityFP,names = ["dst", "similarity"])
        commdf = self._geodf_Map  #to be on safe side we are making a copy
        newCol = "my_similarity_"+str(inComm)
        self._geodf_Map[newCol] = 0
        #print(commdf.columns)
        #print(simdf)
        #print(simdf.shape)
        arr = []   #to extract all dst,<similarity value> pairs 
        for i in range(0,simdf.shape[0]):
            #print("dst i=" + str(i) + " " + str(simdf.iloc[i]['dst']))
            #print("similarity i="+str(i)+" "+str(simdf.iloc[i]['similarity']))
            dst = int(simdf.iloc[i]['dst'])
            sim = float(simdf.iloc[i]['similarity'])
            for it,nb in enumerate(commdf.area_num_1):
                if(int(nb) == int(dst)):
                    commdf.my_similarity[it] = 1.0 
                    #commdf.my_similarity[it] = int(sim*128)


        self._geodf_Map[newCol] = commdf['my_similarity']
        
        return newCol
    
    def getAllCommuns(self,fullrow):
        
        subrow = []
        for row in fullrow:
            #print(row)
            row = row.strip()
            row = row.strip("()")
            #print(row)
            row = row.split(',')
            #print(row)
            subrow.append(row[0])
        #print(subrow)
        return subrow
        
    def fillSimilarityCol2(self,inSimilarityFP):
        
        oAttr = "my_similarity"
        commdf = self._geodf_Map
        fp = open(inSimilarityFP)
        reader = csv.reader(fp)
        #print(sim_df)
        clusterNum = 1
        for row in reader:
            subrow = self.getAllCommuns(row)
            if(len(subrow) == 1):
                continue
            setClustNum = clusterNum
            #setClustNum = 1
            for commun in subrow:
                #print(int(commun))
                commun_nb = int(commun)
                for it,nb in enumerate(commdf.area_num_1):
                    if( int(nb) == commun_nb+1 ):
                        commdf.my_similarity[it] = setClustNum
                        break
                        #commdf.my_similarity[it] = int(sim*128)
                #print("Cluster Number:"+str(clusterNum))
            clusterNum+=1
            #print(clusterNum)
            
        self._geodf_Map['my_similarity'] = commdf['my_similarity']
        return oAttr
    
#Client Code
comm_fp = "../../Data/Static/Boundaries_Community_Areas/geo_export_8ae496a0-1182-4961-8fd2-474ae91317e2.shp"
community = community_map(comm_fp)
#to See the raw data in the object
#attr = ['area_num_1','my_similarity']
#community.showData()

#colMap = "Dark2"
colMap = "Paired"
attr = 'my_similarity'
tmp = attr
#community.plotMap()

years = [2011,2012,2013,2014,2015]
for year in years:
    sim_fp = "../../Data/Total_Data/Output/similarity"+str(year)+".csv"
    tmp = community.fillSimilarityCol2(sim_fp)
    community.plotMap(year,colMap,iAttr=tmp,bShow=False)
    
#print(community._geodf_Map["my_similarity"])

