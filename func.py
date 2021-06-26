import numpy as np 
np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress = True)
import pandas as pd 
import csv
from scipy.stats import pearsonr,kendalltau,spearmanr,ranksums,ttest_rel
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest,SelectPercentile,VarianceThreshold
import mifs
from xgboost import XGBClassifier


##########################feature selection#############################################
def fs_jmi(selected,target):
  df = pd.read_csv('./data/train.csv')
  df_test = pd.read_csv('./data/test.csv')
  X = df.iloc[:,2:].values
  y = df[target]

  X_new = mifs.MutualInformationFeatureSelector(method = 'JMI',n_features = selected).fit(X,y)
  selected_boolean = X_new.support_
  features_list = []
  for i in range (1,5):
    for j in range (1,301):
      features_list.append('S{}_V{}'.format(i,j))
  selected_feature = []
  for i in range (0,len(selected_boolean)):
    if selected_boolean[i] == True:
      selected_feature.append(features_list[i])
  newheader = [target]+selected_feature
  selected_df = df.loc[:,newheader]
  selected_df.to_csv('./rank_data/data_fs_jmi_{}_selected_{}.csv'.format(target,selected),header = newheader,index = None)
  selected_df_test = df_test.loc[:,newheader]
  selected_df_test.to_csv('./test_data/data_fs_jmi_{}_selected_{}.csv'.format(target,selected),header = newheader,index = None)
  with open('./rank_selected_feature/fs_sf_{}_selected_{}.csv'.format(target,selected),'a',newline='')as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['fs_jmi'] + selected_feature)

def fs_mrmr(selected,target):
  df = pd.read_csv('./data/train.csv')
  df_test = pd.read_csv('./data/test.csv')
  X = df.iloc[:,2:].values
  y = df[target]

  X_new = mifs.MutualInformationFeatureSelector(method = 'MRMR',n_features = selected).fit(X,y)
  selected_boolean = X_new.support_
  features_list = []
  for i in range (1,5):
    for j in range (1,301):
      features_list.append('S{}_V{}'.format(i,j))
  selected_feature = []
  for i in range (0,len(selected_boolean)):
    if selected_boolean[i] == True:
      selected_feature.append(features_list[i])
  newheader = [target]+selected_feature
  selected_df = df.loc[:,newheader]
  selected_df.to_csv('./rank_data/data_fs_mrmr_{}_selected_{}.csv'.format(target,selected),
    header = newheader,index = None)
  selected_df_test = df_test.loc[:,newheader]
  selected_df_test.to_csv('./test_data/data_fs_mrmr_{}_selected_{}.csv'.format(target,selected),
    header = newheader,index = None)
  with open('./rank_selected_feature/fs_sf_{}_selected_{}.csv'.format(target,selected),'a',
    newline='')as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['fs_mrmr'] + selected_feature)

def fs_sb(selected,target):
  df = pd.read_csv('./data/train.csv')
  df_test = pd.read_csv('./data/test.csv')
  X = df.iloc[:,2:].values
  y = df[target]

  X_new = SelectKBest(k=selected).fit(X,y)#0.2,0.4,0.8
  selected_boolean = X_new.get_support()
  features_list = []
  for i in range (1,5):
    for j in range (1,301):
      features_list.append('S{}_V{}'.format(i,j))
  selected_feature = []
  for i in range (0,len(selected_boolean)):
    if selected_boolean[i] == True:
      selected_feature.append(features_list[i])
  newheader = [target]+selected_feature
  selected_df = df.loc[:,newheader]
  selected_df.to_csv('./rank_data/data_fs_sb_{}_selected_{}.csv'.format(target,selected),
    header = newheader,index = None)
  selected_df_test = df_test.loc[:,newheader]
  selected_df_test.to_csv('./test_data/data_fs_sb_{}_selected_{}.csv'.format(target,selected),
    header = newheader,index = None)
  with open('./rank_selected_feature/fs_sf_{}_selected_{}.csv'.format(target,selected),'a',
    newline='')as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['fs_sb'] + selected_feature)

def fs_sp(selected,target):
  df = pd.read_csv('./data/train.csv')
  df_test = pd.read_csv('./data/test.csv')
  X = df.iloc[:,2:].values
  y = df[target]

  X_new = SelectPercentile(percentile=selected/1200*100).fit(X,y)
  selected_boolean = X_new.get_support()
  features_list = []
  for i in range (1,5):
    for j in range (1,301):
      features_list.append('S{}_V{}'.format(i,j))
  selected_feature = []
  for i in range (0,len(selected_boolean)):
    if selected_boolean[i] == True:
      selected_feature.append(features_list[i])
  newheader = [target]+selected_feature
  selected_df = df.loc[:,newheader]
  selected_df.to_csv('./rank_data/data_fs_sp_{}_selected_{}.csv'.format(target,selected),
    header = newheader,index = None)
  selected_df_test = df_test.loc[:,newheader]
  selected_df_test.to_csv('./test_data/data_fs_sp_{}_selected_{}.csv'.format(target,selected),
    header = newheader,index = None)
  with open('./rank_selected_feature/fs_sf_{}_selected_{}.csv'.format(target,selected),'a',
    newline='')as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['fs_sp'] + selected_feature)

def fs_w(selected,target):
  df = pd.read_csv('./data/train.csv')
  df_test = pd.read_csv('./data/test.csv')
  pvalue_list = []
  for i in range (2,1202):
    X = df.iloc[:,i].values
    y = df[target]
    xset1 = []
    xset0 = []
    for j in range (0,len(X)):
      if y[j] == 1:
        xset1.append(X[j])
      else:
        xset0.append(X[j])
    score,pvalue = ranksums(xset0,xset1)
    pvalue_list.append(pvalue)
  features_list = []
  for i in range (1,5):
    for j in range (1,301):
      features_list.append('S{}_V{}'.format(i,j))
  selected_feature = []
  pset = []
  for i in range (0,len(pvalue_list)):
    selected_feature.append(features_list[i])
    pset.append(pvalue_list[i])
  rank_pvalue_list = np.array(pd.Series(pvalue_list).rank(method = 'first'))
  rank_pvalue_list = rank_pvalue_list.astype(int)
  rank_selected_feature = []
  for i in range (0,len(rank_pvalue_list)):
    if rank_pvalue_list[i] <= selected:
      rank_selected_feature.append(selected_feature[i])

  newheader = [target]+rank_selected_feature
  selected_df = df.loc[:,newheader]
  selected_df.to_csv('./rank_data/data_fs_w_{}_selected_{}.csv'.format(target,selected),
    header = newheader,index = None)
  selected_df_test = df_test.loc[:,newheader]
  selected_df_test.to_csv('./test_data/data_fs_w_{}_selected_{}.csv'.format(target,selected),
    header = newheader,index = None)
  with open('./rank_selected_feature/fs_sf_{}_selected_{}.csv'.format(target,selected),'a',
    newline='')as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['fs_w'] + rank_selected_feature)

##########################classifier####################################################
from sklearn.svm import SVC
def classifier_svc(x_train,y_train,x_test,y_test,loonumber,dataset):
  classifier = SVC(kernel = 'rbf',probability = True )
  classifier.fit(x_train,y_train)
  y_predict = classifier.predict(x_test)
  prob = classifier.predict_proba(x_test)
  prob = prob.reshape(len(y_predict),2)[:,0]
  result = np.zeros((len(y_predict),3))
  result[:,0] = y_test.reshape(len(y_predict))
  result[:,1] = y_predict.reshape(len(y_predict))
  result[:,2] = prob.reshape(len(y_predict))
  result = pd.DataFrame(result)
  result.to_csv('./prediction_result/classifier_svc_{}.csv'.format(dataset),index = None, 
    header = ['target','predict','prob'])
  
from sklearn.ensemble import AdaBoostClassifier

def classifier_adaboost(x_train,y_train,x_test,y_test,dataset):
  classifier = AdaBoostClassifier()
  classifier.fit(x_train,y_train)
  y_predict = classifier.predict(x_test)
  prob = classifier.predict_proba(x_test)
  prob = prob.reshape(len(y_predict),2)[:,0]
  result = np.zeros((len(y_predict),3))
  result[:,0] = y_test.reshape(len(y_predict))
  result[:,1] = y_predict.reshape(len(y_predict))
  result[:,2] = prob.reshape(len(y_predict))
  result = pd.DataFrame(result)
  result.to_csv('./prediction_result/classifier_adaboost_{}.csv'.format(dataset),index = None, 
    header = ['target','predict','prob'])
  

from sklearn.tree import DecisionTreeClassifier

def classifier_decision_tree(x_train,y_train,x_test,y_test,dataset):
  classifier = DecisionTreeClassifier(min_samples_leaf=10)
  classifier.fit(x_train,y_train)
  y_predict = classifier.predict(x_test)
  prob = classifier.predict_proba(x_test)
  prob = prob.reshape(len(y_predict),2)[:,0]
  result = np.zeros((len(y_predict),3))
  result[:,0] = y_test.reshape(len(y_predict))
  result[:,1] = y_predict.reshape(len(y_predict))
  result[:,2] = prob.reshape(len(y_predict))
  result = pd.DataFrame(result)
  result.to_csv('./prediction_result/classifier_decision_tree_{}.csv'.format(dataset),index = None, 
    header = ['target','predict','prob'])
  
from sklearn.naive_bayes import GaussianNB

def classifier_gaussian_naive_bayesian(x_train,y_train,x_test,y_test,dataset):
  classifier = GaussianNB()
  classifier.fit(x_train,y_train)
  y_predict = classifier.predict(x_test)
  prob = classifier.predict_proba(x_test)
  prob = prob.reshape(len(y_predict),2)[:,0]
  result = np.zeros((len(y_predict),3))
  result[:,0] = y_test.reshape(len(y_predict))
  result[:,1] = y_predict.reshape(len(y_predict))
  result[:,2] = prob.reshape(len(y_predict))
  result = pd.DataFrame(result)
  result.to_csv('./prediction_result/classifier_gaussian_naive_bayesian_{}.csv'.format(dataset),
    index = None, header = ['target','predict','prob'])
  

from sklearn.neighbors import KNeighborsClassifier

def classifier_k_nearest_neighborhood(x_train,y_train,x_test,y_test,dataset):
  classifier = KNeighborsClassifier()
  classifier.fit(x_train,y_train)
  y_predict = classifier.predict(x_test)
  prob = classifier.predict_proba(x_test)
  prob = prob.reshape(len(y_predict),2)[:,0]
  result = np.zeros((len(y_predict),3))
  result[:,0] = y_test.reshape(len(y_predict))
  result[:,1] = y_predict.reshape(len(y_predict))
  result[:,2] = prob.reshape(len(y_predict))
  result = pd.DataFrame(result)
  result.to_csv('./prediction_result/classifier_k_nearest_neighborhood_{}.csv'.format(dataset),
    index = None, header = ['target','predict','prob'])
   
from sklearn.naive_bayes import BernoulliNB

def classifier_bernoulli_naive_bayesian(x_train,y_train,x_test,y_test,dataset):
  classifier = BernoulliNB()
  classifier.fit(x_train,y_train)
  y_predict = classifier.predict(x_test)
  prob = classifier.predict_proba(x_test)
  prob = prob.reshape(len(y_predict),2)[:,0]
  result = np.zeros((len(y_predict),3))
  result[:,0] = y_test.reshape(len(y_predict))
  result[:,1] = y_predict.reshape(len(y_predict))
  result[:,2] = prob.reshape(len(y_predict))
  result = pd.DataFrame(result)
  result.to_csv('./prediction_result/classifier_bernoulli_naive_bayesiann_{}.csv'.format(dataset),
    index = None, header = ['target','predict','prob'])
  
from sklearn.ensemble import RandomForestClassifier

def classifier_random_forest(x_train,y_train,x_test,y_test,dataset):
  classifier = RandomForestClassifier()
  classifier.fit(x_train,y_train)
  y_predict = classifier.predict(x_test)
  prob = classifier.predict_proba(x_test)
  prob = prob.reshape(len(y_predict),2)[:,0]
  result = np.zeros((len(y_predict),3))
  result[:,0] = y_test.reshape(len(y_predict))
  result[:,1] = y_predict.reshape(len(y_predict))
  result[:,2] = prob.reshape(len(y_predict))
  result = pd.DataFrame(result)
  result.to_csv('./prediction_result/classifier_random_forest_{}.csv'.format(dataset),index = None, 
    header = ['target','predict','prob'])
  

from sklearn.linear_model import SGDClassifier

def classifier_stochastic_gradient_descent(x_train,y_train,x_test,y_test,dataset):
  classifier = SGDClassifier(loss = 'modified_huber')
  classifier.fit(x_train,y_train)
  y_predict = classifier.predict(x_test)
  prob = classifier.predict_proba(x_test)
  prob = prob.reshape(len(y_predict),2)[:,0]
  result = np.zeros((len(y_predict),3))
  result[:,0] = y_test.reshape(len(y_predict))
  result[:,1] = y_predict.reshape(len(y_predict))
  result[:,2] = prob.reshape(len(y_predict))
  result = pd.DataFrame(result)
  result.to_csv('./prediction_result/classifier_stochastic_gradient_descent_{}.csv'.format(dataset),
    index = None, header = ['target','predict','prob'])
  

from sklearn.ensemble import BaggingClassifier

def classifier_bagging(x_train,y_train,x_test,y_test,dataset):
  classifier = BaggingClassifier()
  classifier.fit(x_train,y_train)
  y_predict = classifier.predict(x_test)
  prob = classifier.predict_proba(x_test)
  prob = prob.reshape(len(y_predict),2)[:,0]
  result = np.zeros((len(y_predict),3))
  result[:,0] = y_test.reshape(len(y_predict))
  result[:,1] = y_predict.reshape(len(y_predict))
  result[:,2] = prob.reshape(len(y_predict))
  result = pd.DataFrame(result)
  result.to_csv('./prediction_result/classifier_bagging_{}.csv'.format(dataset),index = None, 
    header = ['target','predict','prob'])
  
def classifier_xgboost(x_train,y_train,x_test,y_test,dataset):
  classifier = XGBClassifier()
  classifier.fit(x_train,y_train)
  y_predict = classifier.predict(x_test)
  prob = classifier.predict_proba(x_test)
  prob = prob.reshape(len(y_predict),2)[:,0]
  result = np.zeros((len(y_predict),3))
  result[:,0] = y_test.reshape(len(y_predict))
  result[:,1] = y_predict.reshape(len(y_predict))
  result[:,2] = prob.reshape(len(y_predict))
  result = pd.DataFrame(result)
  result.to_csv('./prediction_result/classifier_xgboost_{}.csv'.format(dataset),index = None, 
    header = ['target','predict','prob'])
  
