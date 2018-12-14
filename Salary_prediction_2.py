
import os
import gc
import time
import psutil
import pandas as pd
import numpy as np
#import pymssql as pmsql
#import pyodbc as podbc
#import sqlalchemy as sqlal
#import pandas.io.sql as psql
import time
#from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import ast
from collections import Counter
import matplotlib.pyplot as plt
import datetime
import pickle

 ## SAVE the variable distributions
def find_And_Save_stribution(x_list,x_str):
    ctr1 = Counter(x_list)
    temp_df = pd.DataFrame.from_dict(ctr1,orient='index').reset_index()
    temp_df = temp_df.rename(columns={'index':x_str,0:'count_1'})
    temp_df = temp_df.assign(Percentage = (temp_df.count_1/temp_df.count_1.sum()))
    fileName = x_str + "_distribution.csv"
    temp_df.to_csv(fileName)
    return temp_df
    
 ## find and return the variable distributions
def find_variable_distr(x_df):
    deg_distr = find_And_Save_stribution(x_df.degree,'degree')
    maj_distr = find_And_Save_stribution(x_df.major,'major')
    companyId_distr = find_And_Save_stribution(x_df.companyId,'companyId')
    industry_distr = find_And_Save_stribution(x_df.industry,'industry')
    jobT_distr = find_And_Save_stribution(x_df.jobType,'jobType')
    return deg_distr,maj_distr,companyId_distr,industry_distr,jobT_distr
    
    
 ## remove invalid rows
def remove_inValid_rows(x_df,x_str):
    if(x_str == 'train'):
         x_df = x_df.loc[x_df.salary >= 0,:]
    x_df = x_df.loc[x_df.milesFromMetropolis >= 0,:]
    x_df = x_df.loc[x_df.yearsExperience >= 0,:]
    return x_df

    

 ## Plot the bar graph to see the initial distribution of different levels of columns
def plot_distributions(d_d,m_d,ci_d,ii_d,job_d):
    d_d.plot(x='degree', y='Percentage', kind='bar')
    m_d.plot(x='major', y='Percentage', kind='bar')
    ci_d.plot(x='companyId', y='Percentage', kind='bar')
    ii_d.plot(x='industry', y='Percentage', kind='bar')
    job_d.plot(x='jobType', y='Percentage', kind='bar')
    
## Plot the box plots of variables to see the distribution
def plotBoxPlots(x_df):
    feature_boxplot_df = x_df.boxplot(column=['yearsExperience', 'milesFromMetropolis'])
    salary_box_plot = x_df.boxplot(column=['salary','yearsExperience', 'milesFromMetropolis'])
    x_df.describe()
    salary_box_plot_Jobtype = x_df.boxplot(column=['salary'],by='jobType')
    salary_box_plot_company = x_df.boxplot(column=['salary'],by='companyId')

    


## Prepare the final Data for prediction
def prepare_Final_test_x(x_df):
    #train_features_salaries_df[['jobId','companyId','degree','major','industry']] = 
    categorical_columns = ['companyId','jobType','degree','major','industry']
    numerical_column = ['yearsExperience','milesFromMetropolis']
    test_features_sal_cat_df = x_df[categorical_columns]
    test_features_sal_num_df = x_df[numerical_column]

    test_X_2_dum = pd.get_dummies(test_features_sal_cat_df)
    train_X = pd.concat([test_X_2_dum.reset_index(drop=True), test_features_sal_num_df.reset_index(drop=True)], axis=1)
    
    return train_X
    



## Splits the main data randomly into train and test data.
def create_train_test_df(x_df):
    #train_features_salaries_df[['jobId','companyId','degree','major','industry']] = 
#    categorical_columns = ['companyId','jobType','degree','major','industry']
    categorical_columns = ['jobType','degree','major','industry']
    numerical_column = ['yearsExperience','milesFromMetropolis']
    train_features_sal_cat_df = x_df[categorical_columns]
    train_features_sal_num_df = x_df[numerical_column]
    train_Y = x_df[['salary']]

    train_X_2_dum = pd.get_dummies(train_features_sal_cat_df)
    train_X = pd.concat([train_X_2_dum.reset_index(drop=True), train_features_sal_num_df.reset_index(drop=True)], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.20, random_state=42)

    return X_train,X_test,y_train,y_test
    
## Calculates and plots variable importance based on Random forest Modelling.
def calculatePlot_Variable_Importance(tr_x,te_x):
    regr = RandomForestRegressor(max_depth=2,random_state=0,n_estimators=100)
    regr.fit(tr_x,te_x.salary.tolist())

    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(tr_x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(tr_x.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
    plt.xticks(range(tr_x.shape[1]), indices)
    plt.xlim([-1, tr_x.shape[1]])
    plt.show()
    

## Does Gradient Boosting modelling for a number of values for Number of Trees so that
## Can find out optimal values for these paramers.
def Actual_Modelling_GB(tr_X,te_X,tr_Y,te_Y,imp_var):
#    var_list = list(tr_X)[[list(tr_X)[i] for i in imp_var]]
    no_trees_List = [25,50,75,100,150,175]
    ntree_mse_list = []
    for this_no in no_trees_List:
        est = GradientBoostingRegressor(n_estimators=this_no, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(tr_X, tr_Y)
        mse_rep = mean_squared_error(te_Y, est.predict(te_X))
        ntree_mse_list.append(mse_rep)
        
    df1 = pd.DataFrame(data={'No_Of_tree':no_trees_List,'MSE':ntree_mse_list})
    return df1

## Does Random forest modelling for a number of values for Number of Trees and Max. Depth so that
## Can find out optimal values for these paramers.
def Actual_Modelling_RF(tr_X,te_X,tr_Y,te_Y,imp_var):
#    var_list = list(tr_X)[[list(tr_X)[i] for i in imp_var]]
    no_trees_List = [25,50,75,100,150,175,200]
    depth_List = [2,4,5,6,10]
    ntree_mse_list = []
    for this_no in no_trees_List:
        regr = RandomForestRegressor(max_depth=2,random_state=0,n_estimators=this_no)
        regr.fit(tr_X,tr_Y.salary.tolist())
        mse_rep_rf = mean_squared_error(te_Y, regr.predict(te_X))
        ntree_mse_list.append(mse_rep_rf)
        
    depth_mse_list = []
    
    for this_depth in depth_List:
        regr_1 = RandomForestRegressor(max_depth=this_depth,random_state=0,n_estimators=100)
        regr_1.fit(tr_X,tr_Y.salary.tolist())
        mse_rep_rf = mean_squared_error(te_Y, regr_1.predict(te_X))
        depth_mse_list.append(mse_rep_rf)
        
        
        
    df1 = pd.DataFrame(data={'No_Of_tree':no_trees_List,'MSE':ntree_mse_list})
    df2 = pd.DataFrame(data={'depth_List':depth_List,'MSE':depth_mse_list})
    
    return df1,df2

## Does neural network modelling for a number of values for Number of hidden layeers and Number of Epoches so that
## Can find out optimal values for these paramers.
def Actual_Modelling_NN(tr_X,te_X,tr_Y,te_Y,imp_var):
#    var_list = list(tr_X)[[list(tr_X)[i] for i in imp_var]]
    print("Modelling NN =====================>>>>>>>>>>>>>>>>>>>>> 1  ")
    hLayer_List = [2,4,5,8,10]
    hLayer_mse_nn_list = []
    
    epoch_List = [100,500,1000,2000]
    epoch_mse_nn_List = []
    
    for this_Layer in hLayer_List:
        print("Modelling NN =====================>>>>>>>>>>>>>>>>>>>>> 2  ")
        # train and test N_p sets of training and test data sets            
        mlp = MLPRegressor(random_state=5, hidden_layer_sizes=[this_Layer], solver='lbfgs', max_iter=1000)
        mlp.fit(tr_X, tr_Y)
#        Error_ave_train = np.mean(0.5 * (y_train - mlp.predict(X_train))**2)
        mse_rep_nn = mean_squared_error(te_Y, mlp.predict(te_X))
        hLayer_mse_nn_list.append(mse_rep_nn)
        
    for this_epoch in epoch_List:
        print("Modelling NN =====================>>>>>>>>>>>>>>>>>>>>> 3  ")
        # train and test N_p sets of training and test data sets            
        mlp = MLPRegressor(random_state=5, hidden_layer_sizes=[5], solver='lbfgs', max_iter=this_epoch)
        mlp.fit(tr_X, tr_Y)
#        Error_ave_train = np.mean(0.5 * (y_train - mlp.predict(X_train))**2)
        mse_rep_nn_epoch = mean_squared_error(te_Y, mlp.predict(te_X))
        epoch_mse_nn_List.append(mse_rep_nn_epoch)
        
    
    df1 = pd.DataFrame(data={'No_Of_Hidden_Layer':hLayer_List,'MSE':hLayer_mse_nn_list})
    df2 = pd.DataFrame(data={'No_Of_Epoches':epoch_List,'MSE':epoch_mse_nn_List})

    
    return df1,df2



    
    
if __name__ == '__main__':
    ## Change to work Directory
    work_project = "C:\\GAUTAM_transfer\\Ohbora_Sans_work\\INDEED_Assignment"
    os.chdir(work_project)
    
    train_features_df = pd.read_csv("C:\\GAUTAM_transfer\\train_features.csv")
    train_salaries_df =  pd.read_csv("C:\\GAUTAM_transfer\\train_salaries.csv")
    
    final_test_features = pd.read_csv("C:\\GAUTAM_transfer\\test_features.csv")
    
    
    
    train_features_salaries_df =train_features_df.merge(train_salaries_df, on = 'jobId',how='inner')
    train_features_salaries_df.to_csv("train_features_salaries_df.csv")
    
    ## Remove any zero salary rows.
    train_features_salaries_df = remove_inValid_rows(train_features_salaries_df,"train")
    
    final_test_features_df = remove_inValid_rows(final_test_features,"test")
    
    
    sal_mean = np.mean(train_features_salaries_df.salary)
    sal_sd = np.std(train_features_salaries_df.salary)
    high_max = (sal_mean + 3 *sal_sd)
    low_min = (sal_mean - 3 * sal_sd)



    wol_train_features_salaries_df = train_features_salaries_df.loc[((low_min < train_features_salaries_df.salary) & (train_features_salaries_df.salary < high_max)),:]
    ol_train_features_salaries_df = train_features_salaries_df.loc[((low_min >= train_features_salaries_df.salary) | (train_features_salaries_df.salary >= high_max)),:]


    ol_train_features_salaries_df[['salary']].hist(bins=10)
    
    ## Distribution of variables withour salary outliers
    wol_deg_d,wol_maj_d,wol_ci_d,wol_ii_d,wol_job_d = find_variable_distr(wol_train_features_salaries_df)
    plot_distributions(wol_deg_d,wol_maj_d,wol_ci_d,wol_ii_d,wol_job_d)

    ## Distribution of variables for salary outliers
    ol_deg_d,ol_maj_d,ol_ci_d,ol_ii_d,ol_job_d = find_variable_distr(ol_train_features_salaries_df)
    plot_distributions(ol_deg_d,ol_maj_d,ol_ci_d,ol_ii_d,ol_job_d)
    
    ## Overll variable importance
    a_train_features_salaries_df = train_features_salaries_df.drop('companyId',axis=1)
    tr_X,te_X,tr_Y,te_Y= create_train_test_df(a_train_features_salaries_df)
    calculatePlot_Variable_Importance(tr_X,tr_Y)

    ## Janitors variable importance
    janitors_train_salary_df = train_features_salaries_df.loc[train_features_salaries_df.jobType == 'JANITOR',:]
    janitors_train_salary_df = janitors_train_salary_df.drop('companyId',axis=1)
    tr_X,te_X,tr_Y,te_Y= create_train_test_df(janitors_train_salary_df)
    calculatePlot_Variable_Importance(tr_X,tr_Y)

    ## Variable importance for jobs CEO,CFO,CTO,VICE_PRESIDENT 
    high_salry_jobTypes = ['CEO','CFO','VICE_PRESIDENT','CTO']
    high_sal_train_sal_df = train_features_salaries_df[train_features_salaries_df['jobType'].isin(high_salry_jobTypes)]
    high_sal_train_sal_df = high_sal_train_sal_df.drop('companyId',axis=1)
    tr_X,te_X,tr_Y,te_Y= create_train_test_df(high_sal_train_sal_df)
    calculatePlot_Variable_Importance(tr_X,tr_Y)

    ## Variable importance for jobs other than CEO,CFO,CTO,VICE_PRESIDENT 
    ok_salry_jobTypes = ['CEO','CFO','VICE_PRESIDENT','CTO']
    ok_sal_train_sal_df = train_features_salaries_df[~train_features_salaries_df['jobType'].isin(high_salry_jobTypes)]
    ok_sal_train_sal_df = ok_sal_train_sal_df.drop('companyId',axis=1)
    tr_X,te_X,tr_Y,te_Y= create_train_test_df(high_sal_train_sal_df)
    calculatePlot_Variable_Importance(tr_X,tr_Y)
    
    
    
    
    ####################################################################
    ##  MODELLING AND TUNING PARAMETERS for 'CEO','CFO','VICE_PRESIDENT','CTO'
    ####################################################################
    high_X_train,high_X_test,high_y_train,high_y_test =create_train_test_df(high_sal_train_sal_df)
    high_imp_features = ['yearsExperience','milesFromMetropolis','major']
    #mse_grb_high = Actual_Modelling_GB(high_X_train,high_X_test,high_y_train,high_y_test,high_imp_features)
    mse_df1,mse_df2 = Actual_Modelling_RF(high_X_train,high_X_test,high_y_train,high_y_test,high_imp_features)
    mse_df1.to_csv("High_sal_RF")

    plt.plot(mse_df1.No_Of_tree,mse_df1.MSE)
    plt.xlabel('No_Of_trees')
    plt.ylabel('Mean_Squared_Error')

    plt.plot(mse_df2.depth_List,mse_df2.MSE)
    plt.xlabel('RF_Depth')
    plt.ylabel('Mean_Squared_Error')


    mse_grb_df = Actual_Modelling_GB(high_X_train,high_X_test,high_y_train,high_y_test,high_imp_features)


    mes_nn_Hl_df,mes_nn_epch_df = Actual_Modelling_NN(high_X_train,high_X_test,high_y_train,high_y_test,high_imp_features)
    plt.plot(mes_nn_Hl_df.No_Of_Hidden_Layer,mes_nn_Hl_df.MSE)
    plt.xlabel('No_Of_Hidden_Layer')
    plt.ylabel('Mean_Squared_Error')
    plt.title('High Salary Salary NT NN')

    plt.plot(mes_nn_epch_df.No_Of_Epoches,mes_nn_epch_df.MSE)
    plt.xlabel('Epoch')
    plt.ylabel('Mean_Squared_Error')
    plt.title('High Salary Salary Epoch NN')
    mes_nn_Hl_df.to_csv("High_mes_nn_Hl_df.csv")
    mes_nn_epch_df.to_csv("mes_nn_epch_df.csv")



   ## Final RF Model using n_estimators = 75 and Max_Depth = 6
   regr_cxo = RandomForestRegressor(max_depth=6,random_state=0,n_estimators=75)
   regr_cxo.fit(high_X_train,high_y_train.salary.tolist())
   mse_rep_rf = mean_squared_error(high_y_test, regr_cxo.predict(high_X_test))


   #########################################################################
   #########################################################################




   #################################################################
   ##  MODELLING AND TUNING PARAMETERS for other than 'CEO','CFO','VICE_PRESIDENT','CTO'
   #################################################################
   ok_X_train,ok_X_test,ok_y_train,ok_y_test =create_train_test_df(ok_sal_train_sal_df)
   ok_imp_features = ['yearsExperience','milesFromMetropolis','major']
   mse_grb_df = Actual_Modelling_GB(ok_X_train,ok_X_test,ok_y_train,ok_y_test,ok_imp_features)


   mse_ok_rf_df1,mse_ok_rf_df2 = Actual_Modelling_RF(ok_X_train,ok_X_test,ok_y_train,ok_y_test,high_imp_features)

   plt.plot(mse_ok_rf_df1.No_Of_tree,mse_ok_rf_df1.MSE)
   plt.xlabel('No_Of_tree')
   plt.ylabel('Mean_Squared_Error')
   plt.title('Ok Salary Random Forest')

   plt.plot(mse_ok_rf_df2.No_Of_tree,mse_ok_rf_df2.MSE)
   plt.xlabel('Max_Depth')
   plt.ylabel('Mean_Squared_Error')
   plt.title('Ok Salary Random Forest-Max_Depth')



   plt.plot(mse_grb_df.No_Of_tree,mse_grb_df.MSE)
   plt.xlabel('No_Of_tree')
   plt.ylabel('Mean_Squared_Error')

   gb_est = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(ok_X_train, ok_y_train)
   mse_rep_ok = mean_squared_error(ok_y_test, gb_est.predict(ok_X_test))




   ok_mes_nn_Hl_df,ok_mes_nn_epch_df = Actual_Modelling_NN(ok_X_train,ok_X_test,ok_y_train,ok_y_test,ok_imp_features)
   plt.plot(ok_mes_nn_Hl_df.No_Of_Hidden_Layer,ok_mes_nn_Hl_df.MSE)
   plt.xlabel('No_Of_Hidden_Layer')
   plt.ylabel('Mean_Squared_Error')
   plt.title('OK Salary Salary NT NN')
   ok_mes_nn_Hl_df.to_csv("ok_mes_nn_Hl_df.csv")

   plt.plot(ok_mes_nn_epch_df.No_Of_Epoches,ok_mes_nn_epch_df.MSE)
   plt.xlabel('Epoch')
   plt.ylabel('Mean_Squared_Error')
   plt.title('OK Salary Salary Epoch NN')
   ok_mes_nn_epch_df.to_csv("ok_mes_nn_epch_df.csv")

   #################################################################
   #################################################################
   
   ########################## Final Neural Network Model OK Salary  ############################
    Ok_salary_mlp_NN = MLPRegressor(random_state=5, hidden_layer_sizes=[5], solver='lbfgs', max_iter=1000)
    Ok_salary_mlp_NN.fit(ok_X_train, ok_y_train)
    #  Error_ave_train = np.mean(0.5 * (y_train - mlp.predict(X_train))**2)
    ok_mse_rep_nn_epoch = mean_squared_error(ok_y_test, Ok_salary_mlp_NN.predict(ok_X_test))
    
    categorical_columns = ['jobType','degree','major','industry']
    numerical_column = ['yearsExperience','milesFromMetropolis']
    
    high_salry_jobTypes = ['CEO','CFO','VICE_PRESIDENT','CTO']
    f_ok_sal_train_sal_df = final_test_features_df[~final_test_features_df['jobType'].isin(high_salry_jobTypes)]
    f_ok_jobIDs = f_ok_sal_train_sal_df[['jobId']]
    
    f_ok_test_features_sal_cat_df = f_ok_sal_train_sal_df[categorical_columns]
    f_ok_test_features_sal_num_df = f_ok_sal_train_sal_df[numerical_column]
    
    
    f_ok_test_X_2_dum = pd.get_dummies(f_ok_test_features_sal_cat_df)
    f_ok_final_test_X = pd.concat([f_ok_test_X_2_dum.reset_index(drop=True), f_ok_test_features_sal_num_df.reset_index(drop=True)], axis=1)
    
     
    f_Ok_sal_pred = Ok_salary_mlp_NN.predict(f_ok_final_test_X)
    ok_job_pred_df = pd.DataFrame(data={'jobId':f_ok_jobIDs.jobId.tolist(),'salary':list(f_Ok_sal_pred)})

#################################################################

########################## Final Neural Network Model High Salary ############################
    high_salary_mlp_NN = MLPRegressor(random_state=5, hidden_layer_sizes=[5], solver='lbfgs', max_iter=1000)
    high_salary_mlp_NN.fit(high_X_train, high_y_train)
    #  Error_ave_train = np.mean(0.5 * (y_train - mlp.predict(X_train))**2)
    high_mse_rep_nn_epoch = mean_squared_error(high_y_test, high_salary_mlp_NN.predict(high_X_test))
    
    categorical_columns = ['jobType','degree','major','industry']
    numerical_column = ['yearsExperience','milesFromMetropolis']
    
    high_salry_jobTypes = ['CEO','CFO','VICE_PRESIDENT','CTO']
    f_high_sal_train_sal_df = final_test_features_df[final_test_features_df['jobType'].isin(high_salry_jobTypes)]
    f_high_jobIDs = f_high_sal_train_sal_df[['jobId']]
    
    f_high_test_features_sal_cat_df = f_high_sal_train_sal_df[categorical_columns]
    f_high_test_features_sal_num_df = f_high_sal_train_sal_df[numerical_column]
    
    
    f_high_test_X_2_dum = pd.get_dummies(f_high_test_features_sal_cat_df)
    f_high_final_test_X = pd.concat([f_high_test_X_2_dum.reset_index(drop=True), f_high_test_features_sal_num_df.reset_index(drop=True)], axis=1)
    
     
    f_high_sal_pred = Ok_salary_mlp_NN.predict(f_high_final_test_X)
    high_job_pred_df = pd.DataFrame(data={'jobId':f_high_jobIDs.jobId.tolist(),'salary':list(f_high_sal_pred)})
    
    final_job_sal_pred_df = ok_job_pred_df.append(high_job_pred_df)
#################################################################
    
    
    




