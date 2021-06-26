import numpy as np 
np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress = True)
import pandas as pd 
import csv
from func import *
from sklearn.model_selection import LeaveOneOut
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

for n_feature in range (1,21):
	for fs in ['jmi','mrmr','skb','sp','w']:
		for taget in ['dd','lnm']:
			item = n_feature*5
			df = pd.read_csv('./rank_data/data_fs_{}_{}_selected_{}.csv'.format(fs,target,item))
			dataset_name = 'data_fs_{}_{}_selected_{}'.format(fs,target,item)
			x_train = df.iloc[:,2:].values
			y_train = df[taget]
			
			
			df_test = pd.read_csv('./test_data/data_fs_{}_{}_selected_{}.csv'.format(fs,target,item))
			x_test = df_test.iloc[:,2:].values
			y_test = df_test[taget]
			
			classifier_svmc_rbf(x_train,y_train,x_test,y_test,dataset = dataset_name)
			classifier_adaboost(x_train,y_train,x_test,y_test,dataset = dataset_name)
			classifier_decision_tree(x_train,y_train,x_test,y_test,dataset = dataset_name)
			classifier_gaussian_naive_bayesian(x_train,y_train,x_test,y_test,dataset = dataset_name)
			classifier_k_nearest_neighborhood(x_train,y_train,x_test,y_test,dataset = dataset_name)
			classifier_logisitic_regression(x_train,y_train,x_test,y_test,dataset = dataset_name)
			classifier_bernoulli_naive_bayesian(x_train,y_train,x_test,y_test,dataset = dataset_name)
			classifier_random_forest(x_train,y_train,x_test,y_test,dataset = dataset_name)
			classifier_stochastic_gradient_descent(x_train,y_train,x_test,y_test,dataset = dataset_name)
			classifier_bagging(x_train,y_train,x_test,y_test,dataset = dataset_name)
			classifier_xgboost(x_train,y_train,x_test,y_test,dataset = dataset_name)