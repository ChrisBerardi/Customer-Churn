"""
Author: Chris Berardi
Data Cleaning, Exploration, Modeling and Validation of customer churn data
October-November 2018
"""
import pandas as pd
import numpy as np

file_path = "C:/Users/Saistout/Desktop/data/customer churn/"
df = pd.read_csv(file_path + 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

#Basic Data Exploration
print(df.head())

print(df.describe(include='all'))

print(df.count())
print(df['Churn'].value_counts())

