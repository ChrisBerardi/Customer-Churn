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

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

# Function for calculating loss and confusion matrix
def loss_cal(y, y_predict, fp_cost, fn_cost, tp_cost, display=True):
    loss     = [0, 0, 0, 0]       #False Neg Cost, False Pos Cost, 
                                  #True Neg Cost, True positive savings
    conf_mat = [0, 0, 0, 0] #tn, fp, fn, tp
    for j in range(len(y)):
        if y[j]==0:
            if y_predict[j]==0:
                conf_mat[0] += 1 #True Negative
            else:
                conf_mat[1] += 1 #False Positive
                loss[1] += fp_cost[j]
        else:
            if y_predict[j]==1:
                conf_mat[3] += 1 #True Positive
                loss[3] += fn_cost[j]
                loss[2] += tp_cost[j]
            else:
                conf_mat[2] += 1 #False Negative
                loss[0] += fn_cost[j]
    if display:
        fn_loss = loss[0]
        fp_loss = loss[1]
        tp_loss = loss[2]
        tp_save = loss[3]
        total_loss = fn_loss + fp_loss + tp_loss
        total_saved = tp_save-total_loss
        misc    = conf_mat[1] + conf_mat[2]
        misc    = misc/len(y)
        print("{:.<23s}{:10.4f}".format("Misclassification Rate", misc))
        print("{:.<23s}{:10.0f}".format("False Negative Cost", fn_loss))
        print("{:.<23s}{:10.0f}".format("False Positive Cost", fp_loss))
        print("{:.<23s}{:10.0f}".format("True Positive Cost", tp_loss))
        print("{:.<23s}{:10.0f}".format("Total Loss", total_loss))
        print("{:.<23s}{:10.0f}".format("Total Saved", tp_save))
        print("{:.<23s}{:10.0f}".format("Net Saved", total_saved))
    return loss, conf_mat


#Define number of loops for optimization
its = 1000    

file_path = "C:/Users/Saistout/Desktop/data/customer churn/"
df = pd.read_csv(file_path + 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

#Drop the CustomerID attribute
df=df.iloc[:,1:len(df.columns)]

#Basic Data Exploration
print(df.head())

print(df.describe(include='all'))

for i in (0, len(df.columns)-1):
    print(df.iloc[:,i].value_counts())
    
#Calculate total amount lost to churn
index = df['Churn'] == 'Yes'
total_churn = df['MonthlyCharges'][index].sum()

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
#lose and retained functiosn to have any hope of modeling this data
#Lose will be the montly payments lost i.e. churn not predicted plus the cost
#of preventing churn i.e. cost per true negative or false negative
#Retained will be the monthly payments for true negatives i.e. payments from
#churn prevented 

# Setup false positive and false negative costs for each transaction
 #Make a nominal cost for a fale positive i.e. they get thing even though
 #They won't churn
fp_cost = np.array(df['MonthlyCharges']*.2)
#False negatives cost the full amount i.e. they churn and we lose them
fn_cost = np.array(df['MonthlyCharges']) 
#Nominal true positive cost i.e. cost to run program to stop churn for 
#those that will
tp_cost = np.array(df['MonthlyCharges']*.2)

# Setup random number seeds
rand_val = np.array([1,5, 10, 15, 168, 21, 5156, 71686])
rand_val = np.array(range(0,its))
# Ratios of Majority:Minority Events
ratio = ['30:70','40:60', '50:50', '60:40', '70:30']
# Dictionaries contains number of minority and majority events in each ratio sample
# n_majority = ratio x n_minority
rus_ratio = ({0:801,1:1869},{0:1246,1:1869},{0:1869, 1:1869}, {0:2804, 1:1869}, \
             {0:4361, 1:1869})

# Best model is one that maximizes savings
#Run 1000 times to get average saved
min_saved   = -9e+15
best_ratio = 0
for k in range(len(rus_ratio)):
    rand_vals = (k+1)*rand_val
    print("\nDecision Tree Model using " + ratio[k] + " RUS")
    fn_loss  = np.zeros(len(rand_vals))
    fp_loss  = np.zeros(len(rand_vals))
    misc     = np.zeros(len(rand_vals))
    tp_loss  = np.zeros(len(rand_vals))
    tp_saved = np.zeros(len(rand_vals))
    for i in range(len(rand_vals)):
        rus = RandomUnderSampler(ratio=rus_ratio[k], \
                random_state=rand_vals[i], return_indices=False, \
                replacement=False)
        X_rus, y_rus = rus.fit_sample(X_t, np_y_t)
        tr = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, \
                                 min_samples_split=5,criterion='gini')
        tr.fit(X_rus, y_rus)
        loss, conf_mat = loss_cal(np_y_t, tr.predict(X_t), fp_cost, fn_cost,\
                                  tp_cost,display=False)
        fn_loss[i]  = loss[0]
        fp_loss[i]  = loss[1]
        tp_loss[i]  = loss[2]
        tp_saved[i] = loss[3]
        misc[i]    = conf_mat[1] + conf_mat[2]
    misc = np.sum(misc)/(len(df) * len(rand_vals))
    fn_avg_loss = np.average(fn_loss)
    fp_avg_loss = np.average(fp_loss)
    tp_avg_loss = np.average(tp_loss)
    tp_avg_saved   = np.average(tp_saved)
    total_loss  = fn_loss + fp_loss + tp_loss
    net_saved   = tp_saved - total_loss
    avg_loss    = np.average(total_loss)
    std_loss    = np.std(total_loss)
    avg_saved   = np.average(net_saved)
    std_saved   = np.std(net_saved)
    print("{:.<23s}{:10.4f}".format("Misclassification Rate", misc))
    print("{:.<23s}{:10.0f}".format("False Negative Cost", fn_avg_loss))
    print("{:.<23s}{:10.0f}".format("False Positive Cost", fp_avg_loss))
    print("{:.<23s}{:10.0f}".format("True Positive Cost", tp_avg_loss))
    print("{:.<23s}{:10.0f}".format("True Positive Savings", tp_avg_saved))
    print("{:.<23s}{:10.0f}{:5s}{:<10.2f}".format("Total Loss", avg_loss, \
                  " +/- ", std_loss))
    print("{:.<23s}{:10.0f}{:5s}{:<10.2f}".format("Net Saved", avg_saved, \
                  " +/- ", std_saved))
    if avg_saved > min_saved:
        min_saved   = avg_saved
        best_ratio = k
print("Optimum Ratio is: ", ratio[best_ratio])
#Implement hyperparameter optimization for decision tree using 50/50 RUS
#Optimize the depth of the trees
#Use savings to determine best model
#Run each 1000 times to generate average and sd of saved
depth_list = [5, 6, 7, 8, 10, 12]
min_saved   = -9e+15
best_ratio = 0
for d in depth_list:
    saved = np.zeros(len(rand_vals))
    for i in range(len(rand_vals)):
        dtc = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5, \
                                     min_samples_split=5)
        dtc = dtc.fit(X_t,y_t)
        rus = RandomUnderSampler(ratio=rus_ratio[best_ratio], \
                                 random_state=rand_vals[i], return_indices=False, \
                                 replacement=False)
        X_rus, y_rus = rus.fit_sample(X_t, np_y_t)
        tr = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, \
                                 min_samples_split=5,criterion='gini')
        tr.fit(X_rus, y_rus)
        loss, conf_mat = loss_cal(np_y_t, tr.predict(X_t), fp_cost, fn_cost,\
                                      tp_cost,display=False)
        saved[i] = loss[3]-loss[0]-loss[1]-loss[2]
    print("\nTree Depth: ", d)
    print("{:.<23s}{:10.4f}".format("Average Savings", np.average(saved)))
    print("{:.<23s}{:10.4f}".format("Sd Savings", np.std(saved)))

    if np.average(saved) > min_saved:
        min_saved = np.average(saved)
        best_depth = d
print("\nBest based on Highest Savings")
print("Best Depth = ", best_depth)


# Ensemble Modeling - Averaging Classification Probabilities
avg_prob = np.zeros((len(np_y_t),2))
# Setup 10 random number seeds for use in creating random samples
np.random.seed(12345)
max_seed = 2**31 - 1
rand_value = np.random.randint(1, high=max_seed, size=1000)
# Model 100 random samples, each with a 50:50 ratio
for i in range(len(rand_value)):
    rus = RandomUnderSampler(ratio=rus_ratio[best_ratio], \
                    random_state=rand_value[i], return_indices=False, \
                    replacement=False)
    X_rus, y_rus = rus.fit_sample(X_t, np_y_t)
    ltr = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=5, \
                                 min_samples_split=5,criterion='gini')
    tr.fit(X_rus, y_rus)
    avg_prob += tr.predict_proba(X_t)
print(tr.feature_importances_)
avg_prob = avg_prob/len(rand_value)
# Set y_pred equal to the predicted classification
y_pred = avg_prob[0:,0] < 0.5
y_pred.astype(np.int)
# Calculate loss from using the ensemble predictions
print("\nEnsemble Estimates based on averaging",len(rand_value), \
      "Models with depth",best_depth)
loss, conf_mat = loss_cal(np_y_t, y_pred,fp_cost,fn_cost, tp_cost)
