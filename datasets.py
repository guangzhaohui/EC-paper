'''
Data set split
'''

import numpy as np
import pandas as pd
import random

df = pd.read_csv('./data/dd2_im2.csv')
header_list = df.columns
all_index = []
for i in range (0,len(df)):
	all_index.append(i)
index_temp = all_index.copy()
random.shuffle(index_temp)
test_index = index_temp[int(len(df)*0.8):]
train_index = index_temp[:int(len(df)*0.8)]
print(test_index,train_index)
test_data = []
for i in range (0,len(test_index)):
	test_data.append(np.array(df)[test_index[i]])

train_data = []
for i in range (0,len(train_index)):
	train_data.append(np.array(df)[train_index[i]])

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

train_data.to_csv('./data/train.csv', header = header_list, index = None)
test_data.to_csv('./data/test.csv',header = header_list, index = None)