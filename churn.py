"""
Author: Chris Berardi
Data Cleaning, Exploration, Modeling and Validation of customer churn data
October-November 2018
"""
import pandas as pd
import numpy as np

from Class_replace_impute_encode import ReplaceImputeEncode

from Class_regression import logreg
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

from Class_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

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

#TotalCharges is a str, we need it to be an int for a lot of reasons
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


#Logistics Regression
max_f1 = 0
score_list = ['accuracy', 'recall', 'precision', 'f1']

#Encode for logistic regressions
rie_l = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', 
                          interval_scale = 'std', drop=True, display=True)
encoded_df_l = rie_l.fit_transform(df)
X_l = encoded_df_l.drop(target, axis=1)
y_l = encoded_df_l[target]
np_y_l = np.ravel(y_l) #convert dataframe column to flat array


#Do feature selection using random forest classifiers to determine which
#predictors to include in the logistic regression
features = ExtraTreesClassifier(n_estimators=500)
features.fit(X_l,np_y_l)
print(features.feature_importances_)
#Only the interval predictors are important
#Try two logistic models: one with all predictors, one with only the top 3 
#predictors
X_l_pred = X_l[['tenure', 'MonthlyCharges', 'TotalCharges']]


#Set up arrays for testing
X_train_l, X_validate_l, y_train_l, y_validate_l = \
            train_test_split(X_l,y_l,test_size = 0.3, random_state=12345)
np_y_validate_l = np.ravel(y_validate_l)
np_y_train_l = np.ravel(y_train_l)

X_train_l, X_validate_l, y_train_l, y_validate_l = \
            train_test_split(X_l_pred,y_l,test_size = 0.3, random_state=12345)
np_y_validate_l = np.ravel(y_validate_l)
np_y_train_l = np.ravel(y_train_l)

#Full Model
print("\nFull Model: ")
lgr = LogisticRegression()
lgr.fit(X_l, np_y_l)
scores = cross_validate(lgr, X_l,np_y_l,\
                        scoring=score_list, return_train_score=False, \
                        cv=10)
print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
for s in score_list:
    var = "test_"+s
    mean = scores[var].mean()
    std  = scores[var].std()
    print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))

lgc = LogisticRegression()
lgc = lgc.fit(X_train_l,np_y_train_l)

logreg.display_binary_split_metrics(lgc, X_train_l, np_y_train_l, X_validate_l\
                                     , np_y_validate_l)

#Selected Variables
print("\nTop 3 Predictors: ")
lgr = LogisticRegression()
lgr.fit(X_l_pred, np_y_l)
scores = cross_validate(lgr, X_l_pred,np_y_l,\
                        scoring=score_list, return_train_score=False, \
                        cv=10)
print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
for s in score_list:
    var = "test_"+s
    mean = scores[var].mean()
    std  = scores[var].std()
    print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))

lgc = LogisticRegression()
lgc = lgc.fit(X_train_l,np_y_train_l)

logreg.display_binary_split_metrics(lgc, X_train_l, np_y_train_l, X_validate_l\
                                     , np_y_validate_l)

#Not the greatest results, maybe decision tree will provide better results
#Decision Tree

#Tree encoding
rie_t = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', \
                           drop=False,interval_scale = None, display=True)
encoded_df_t = rie_t.fit_transform(df)
X_t = encoded_df_t.drop(target, axis=1)
y_t = encoded_df_t[target]
np_y_t = np.ravel(y_t) #convert dataframe column to flat array

X_train_t, X_validate_t, y_train_t, y_validate_t = \
            train_test_split(X_t, np_y_t,test_size = 0.3, random_state=12345)

#Decision Tree Models
# Cross Validation
depth_list = [3, 4, 5, 6, 7, 8, 10, 12]
max_f1 = 0
for d in depth_list:
    print("\nMaximum Tree Depth: ", d)
    dtc = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5, \
                                 min_samples_split=5)
    dtc = dtc.fit(X_t,y_t)
    scores = cross_validate(dtc, X_t, y_t, scoring=score_list, \
                            return_train_score=False, cv=10)
    
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        var = "test_"+s
        mean = scores[var].mean()
        std  = scores[var].std()
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        if mean > max_f1:
            max_f1 = mean
            best_depth    = d
            
print("\nBest based on F1-Score")
print("Best Depth = ", best_depth)
# Evaluate the tree with the best depth
dtc = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=5, min_samples_split=5)
dtc = dtc.fit(X_train_t,y_train_t)

print("\nDecision Tree")
print("\nDepth",best_depth)
DecisionTree.display_binary_split_metrics(dtc, X_train_t, y_train_t, \
                                     X_validate_t, y_validate_t)

#Huge false positive rate! We are missing most of the churn, need to move on to
#lose function to have any hope of modeling this data
