"""
Author: Chris Berardi
Data Cleaning, Exploration, Modeling and Validation of customer churn data
October-November 2018
"""
import pandas as pd
import numpy as np
from Class_replace_impute_encode import ReplaceImputeEncode
from sklearn.linear_model import LogisticRegression

file_path = "C:/Users/Saistout/Desktop/data/customer churn/"
df = pd.read_csv(file_path + 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

#Drop the CustomerID attribute
df=df.iloc[:,1:len(df.columns)]

#Basic Data Exploration
print(df.head())

print(df.describe(include='all'))

for i in (0, len(df.columns)-1):
    print(df.iloc[:,i].value_counts())

#Change all null values for TotalCharges to zero to indicate they haven't
#had any yet.
index = df['TotalCharges'] == ' '
df['TotalCharges'].loc[index] = '0'

#TotalCharges is a str, we need it to be an int
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# Attribute Map:  the key is the name in the DataFrame
# The first number of 0=Interval, 1=Binary, 2=Nominal
# The 1st tuple for interval attributes is their lower and upper bounds
# The 1st tuple for categorical attributes is their allowed categories
# The 2nd tuple contains the number missing and number of outliers
attribute_map = {
    'gender'           :[1,('Female','Male'),[0,0]], #1=Female
    'SeniorCitizen'    :[1,(0,1),[0,0]],
    'Partner'          :[1,('Yes','No'),[0,0]],
    'Dependents'       :[1,('Yes','No'),[0,0]],
    'tenure'           :[0,(0,72),[0,0]],
    'PhoneService'     :[1,('Yes','No'),[0,0]],
    'MultipleLines'    :[2,('No','Yes','No phone service'),[0,0]],
    'InternetService'  :[2,('DSL','No','Fiber optic'),[0,0]],
    'OnlineSecurity'   :[2,('Yes','No', 'No internet service'),[0,0]],
    'OnlineBackup'     :[2,('Yes','No', 'No internet service'),[0,0]],
    'DeviceProtection' :[2,('Yes','No', 'No internet service'),[0,0]],
    'TechSupport'      :[2,('Yes','No', 'No internet service'),[0,0]],
    'StreamingTV'      :[2,('Yes','No', 'No internet service'),[0,0]],
    'StreamingMovies'  :[2,('Yes','No', 'No internet service'),[0,0]],
    'Contract'         :[2,('Month-to-month','Two year','One year'),[0,0]],
    'PaperlessBilling' :[1,('Yes','No'),[0,0]],
    'PaymentMethod'    :[2,('Electronic check','Mailed check',
                            'Bank transfer (automatic)',
                            'Credit card (automatic)'),[0,0]],
    'MonthlyCharges'   :[0,(18.25,188.75),[0,0]],
    'TotalCharges'     :[0,(0,8700),[0,0]], #int as str hides max value!
    'Churn'            :[1,('Yes','No'),[0,0]],
}

#Define the target
target = ['Churn']

#Encode for logistic regression
rie_l = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', 
                          interval_scale = 'std', drop=True, display=True)
encoded_df_l = rie_l.fit_transform(df)
X_l = encoded_df_l.drop(target, axis=1)
y_l = encoded_df_l[target]
np_y_l = np.ravel(y_l) #convert dataframe column to flat array
