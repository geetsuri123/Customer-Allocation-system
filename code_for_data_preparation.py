#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:19:48 2023

@author: liyuxuan
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import openpyxl

#import original file 
file_population = "ukpopestimatesmid2021on2021geographyfinal.xls"
file_district = "Local_Authority_Districts.csv"

df_population_estimates = pd.read_excel(file_population,sheet_name="MYE2 - Persons", header=7)
df_district = pd.read_csv(file_district)

#rename df_district LAD22CD column to use as key in merge
df_district.rename({'LAD22CD' : 'Code'}, axis='columns',inplace=True)

#merge two file
df_unique_customer = pd.merge(df_district, df_population_estimates, on='Code', how = 'left')


#filter out useful column 
df_unique_customer_filtered = df_unique_customer[['OBJECTID','Code','LAD22NM','LONG','LAT','Name','Geography','All ages']]


# Replace "All ages" column with "Population"
df_unique_customer_filtered ['Population'] = df_unique_customer_filtered ['All ages']
df_unique_customer_filtered.drop('All ages', axis=1, inplace=True)
df_unique_customer_filtered ['Customers'] = df_unique_customer_filtered ['LAD22NM']


# Add demand for customers with value population multiplied by 0.1%
df_unique_customer_filtered['Demand'] = df_unique_customer_filtered['Population'] * 0.001

#filter out customer name and demand

customer_demand = df_unique_customer_filtered[['Customers','Demand']]


#-------------------------------------------------------------------------------------------------------------------------

#calculating distance matrix
# read in CSV file containing latitude and longitude values

customer = df_unique_customer_filtered
facility=pd.read_excel("facilities_geo.xlsx")


# extract latitude and longitude values for each point
latitudes = customer["LAT"]
longitudes = customer["LONG"]
local_authority=customer["Customers"]


latitudes2 = facility["LAT"]
longitudes2 = facility["LONG"]
facility_name = facility["Facility"]



# create a matrix of latitude and longitude values
points1 = list(zip(customer["LAT"],customer["LONG"]))
points2 = list(zip(facility["LAT"], facility["LONG"]))

# function to calculate distance between two points
def distance(lat1, long1, lat2, long2):
    # convert degrees to radians
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])

    # calculate differences in latitude and longitude
    dlat = lat2 - lat1
    dlong = long2 - long1

    # calculate Haversine formula
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    R = 6371 # radius of the Earth in km
    d = R * c

    return d



# calculate distances between all pairs of points
dists = []
for i in range(len(points1)):
    for j in range(len(points2)):
        d = distance(points1[i][0], points1[i][1], points2[j][0], points2[j][1])
        dists.append(d)
        
# create a distance matrix with city names as row and column labels
matrix_size = len(points1)
distance_matrix = pd.DataFrame(0, index=local_authority, columns=facility_name)
for i in range(matrix_size):
    for j in range(len(points2)):
        distance_matrix.iloc[i, j] = dists[i * len(points2) + j]

# display distance matrix
print(distance_matrix)

#---------------------------------------------------------------------------------
#retrieve data and store in xlsx

# create an Excel writer object
final= pd.ExcelWriter('prepared_data_result.xlsx', engine='xlsxwriter')

# write each dataframe to a different tab in the Excel file
distance_matrix.to_excel(final,sheet_name='Distance')
customer_demand.to_excel(final,sheet_name='Demand')


# save the Excel file
final.save()











