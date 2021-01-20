"""
Implementation of Logistic Regression Algorithm in classifying diseased trees
+ Normalize dataset for training process and classifying process as well
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing

# scale larger positive and values to between -1,1 depending on the largest
# value in the data
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))


# Check the scenario when merging two sets of data into one set, and randomly pick
df_merged = pd.read_csv("training-merged.csv", header=0)

# convert output to numeric ones (n -> 0.0, w -> 1.0)
for i in range(df_merged.shape[0]):
	if(df_merged.iloc[i,0] == 'n'):
		df_merged.iloc[i,0] = 0.0
	else: df_merged.iloc[i,0] = 1.0

# convert column class from char -> float
df_merged[['class']] = df_merged[['class']].astype('float')

# split training inputs and output
X_merged = df_merged.iloc[:,1:]
X_merged = np.array(X_merged)
X_merged = min_max_scaler.fit_transform(X_merged)
Y_merged = df_merged.iloc[:,0]
Y_merged = np.array(Y_merged)

# save the normalized training dataset
X_merged = pd.DataFrame.from_records(X_merged,columns=df_merged.columns[1:])
X_merged.insert(0,'class',Y_merged)
X_merged.to_csv('normalized_merged_set.csv', index=False)