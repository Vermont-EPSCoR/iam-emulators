
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

pd.set_option('display.max_columns', None)



root_path = Path("Project Directory")
data_path = root_path / 'comboCSV.csv'


# a few helper functions
def concentration_to_class(conc):
	if conc < 2.6:
		return 0
	elif conc < 10:
		return 1
	elif conc < 40:
		return 2
	else:
		return 3

def StringToMultipleBinary(dfColumn):
    uniqueVals = dfColumn.unique()
    newCols = pd.DataFrame()
    for value in uniqueVals:
        newCols[value] = dfColumn.apply(lambda row: 1 if row == value else 0)
    return newCols


csv_data = pd.read_csv(data_path, low_memory=False)

# lets repeat the process as in the original Random-forest.py file. We will ignore some of the variabels 
base_ignored_vars = [
    'year', 'month', 'day', 'decade', 'season', 'abm', 'CHLAVESurfaceMax', 
    'TPSurfaceMax', 'TPSurfaceMean', 'SecchiSurfaceMax', 'SecchiSurfaceMean', 
    'TEMPSurfaceMax', 'TEMPSurfaceMean', 'CHLAVESurfaceMean', 'IAMRun'
]


# lets filter out some of the data
csv_data_filt = csv_data[(csv_data['month'] >= 4) & (csv_data['month'] <= 11)].reset_index(drop=True)
csv_data_year = csv_data_filt[(csv_data_filt['year'] <= 2047)].reset_index(drop=True)


# next - lets define the class variable - we will use concentration_to_class function
csv_data_year['CHLAClass'] = csv_data_year['CHLAVESurfaceMean'].apply(concentration_to_class) # for regression comment this line and change the name of the CSVs

# lets drop the columns
csv_data_filtered = csv_data_year.drop(columns=base_ignored_vars)
print(csv_data_filtered.columns)


# next -  we will one hot encode some of the strings columns
csv_data_one_hot_encode = csv_data_filtered.copy()
for stringVar in ['gcm', 'rcp', 'reduxP']:
    one_hot = StringToMultipleBinary(csv_data_one_hot_encode[stringVar])
    csv_data_one_hot_encode = pd.concat([csv_data_one_hot_encode.drop(columns=[stringVar]), one_hot], axis=1)

# at this moment our dataframe is ready to be used for modeling
df = csv_data_one_hot_encode.copy()

# split it into training - testing
df_train, df_test = train_test_split(df, test_size=0.20)

# lets also save the data as CSVs so later we dont have to redo the whole process
outpath_train = root_path / 'comboCSV_train.csv'
outpath_test = root_path / 'comboCSV_test.csv'

df_train.to_csv(outpath_train, index=False)
df_test.to_csv(outpath_test, index=False)