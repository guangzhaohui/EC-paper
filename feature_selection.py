import numpy as np 
np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress = True)
import pandas as pd 
import csv
from scipy.stats import pearsonr,kendalltau,spearmanr,ranksums,ttest_rel
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest,SelectPercentile,VarianceThreshold
import mifs
from sklearn.externals import joblib
from xgboost import XGBClassifier
from selected_anyfeatures4 import * 
import numpy as np 
np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress = True)
import pandas as pd 
import csv
from func import *
from sklearn.model_selection import LeaveOneOut
import os
gen_folder('./rank_data')
gen_folder('./rank_selected_feature')
gen_folder('./test_data')
gen_folder('./prediction_result')
gen_folder('./metrics')
for i in range (1,21):
	print(i/20)
	selected = 5*i
	for target in ['dd','lnm']:

		fs_sb(selected,target)

		fs_sp(selected,target)

		fs_w(selected,target)

		fs_jmi(selected,target)

		fs_mrmr(selected,target)