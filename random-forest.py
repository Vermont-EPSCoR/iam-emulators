# Gitlab site is great! - https://github.com/parrt/random-forest-importances

import re
import glob
import os
from rfpimp import *
from scenario_configuration import IAM_CONFIGURATION as settings
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

scenario_regex = re.compile(''.join([
        r'^(?:(?P<decade>[0-9]{4})-)?',
        r'(?P<abm>.+?)-',
        r'(?P<gcm>.+?)-',
        r'(?P<rcp>(?:rcp)?[0-9]{1,3}|NORCP)',
        r'(?:-(?P<p_redux>[A-Za-z0-9]+_redux))?',
        r'-landuse_proportions\.R'
    ]))


def StringToNum(df, numSpacing = 10):
  return df.replace(to_replace=df.unique(), value=range(0,len(df.unique())*numSpacing,numSpacing))


def StringToMultipleBinary(dfColumn):
  uniqueVals = dfColumn.unique()
  newCols = pd.DataFrame()
  for value in uniqueVals:
    newCols[value] = dfColumn.apply(lambda row: 1 if row == value else 0)
  return newCols


def DetermineReduction(reduxScenario, year):
  ## This might be better if I sorted the keys first and then broke out of the loop when dictYear > year (would have to keep last year in variable)
  
  p_dict = settings['P_REDUCTION_SCENARIOS'][reduxScenario]
  
  reduction = 0
  lastYear = 0
  for dictYear in p_dict.keys():
    if dictYear <= year and dictYear >= lastYear:
      reduction = p_dict[dictYear]
      lastYear = dictYear
  return reduction


#########
# Main Code
#########

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print("Loading file...")
# Use nrows parameter to limit size (104000 = 2031)
# df = pd.read_csv('/raid/pclemins/random-forest/sediment_tabular_data_apr-nov_until_2050_includes_TSIchl_with_inputs_TP_Reductions_LandUse.csv')
df = pd.read_csv('/raid/pclemins/IAMData-P_Reduction_V2/sediment_tabular_data.csv')

print(f'Imported csv size: {df.shape}')

print('Filtering to include only Apr - Nov...')
df = df[(df['month'] >= 4) & (df['month'] <= 11)]
print(f'Filtered csv size: {df.shape}')
df = df[(df['year'] <= 2047)]
print(f'Filtered csv size: {df.shape}')

print("Loading landuse_proportion files...")
landuseDF = pd.DataFrame()
for file in glob.glob('/epscorfs/iam-workspace/pclemins/workflow/P_Reduction_V2/outputs/*landuse_proportions.R'):
  result = re.search(scenario_regex,os.path.basename(file)).groupdict()
  scenario = result['gcm']+'.'+result['rcp']+'.'+result['p_redux']+'.'+result['abm']
  with open(file, "r") as openFile:
    for line in openFile:
      lineparts = line.split("=")
      if len(lineparts) == 2:
        landuseDF.loc[result['decade']+'.'+scenario, lineparts[0].strip()] = lineparts[1].strip()

print("Adding abm landuse proportion columns...")
for luType in ['undev','dev','ag']:
  print("  "+luType+"...")
  df['abm-'+luType] = df.apply(lambda row: landuseDF.loc[str(
    row['decade']) + '.' +
    row['gcm'] + '.' +
    row['rcp'] + '.' +
    row['reduxP'] + '.' +
    row['abm'],
    'proportion_'+luType], axis=1)

# print("Calculating phosphorus reduction for each row...")
# df['reduction'] = df.apply(lambda row: DetermineReduction(row['redux'], row['year']), axis=1)
# for TPcolumn in ['missisquoi_TP','pike_TP','rock_TP','basin_TP_gps']:
#   print(f"Applying phosphorus reduction to {TPcolumn}...")
#   df[f'{TPcolumn}_PreReduction'] = df[TPcolumn]
#   df[TPcolumn] = df[TPcolumn] * ((100-df['reduction'])/100)

# print(df[['year', 'scenario', 'reduction']][-10:])
# print(df['missisquoi_TP_PreReduction'][-10:])
# print(df['missisquoi_TP'][-10:])

# print("Replacing Strings with Numbers...")
# for stringVar in ['abm', 'gcm', 'rcp', 'redux']:
#   df[stringVar] = StringToNum(df[stringVar])

print("Replacing Strings with Multiple 0/1 Columns...")
for stringVar in ['gcm', 'rcp', 'reduxP']:
  df = pd.concat([df, StringToMultipleBinary(df[stringVar])], axis=1)
  df = df.drop(stringVar, axis=1)
  
print("Analysis DataFrame Columns:")
print(df.columns)
df[df.isnull().any(axis=1)].to_csv("nan.csv")

df_train, df_test = train_test_split(df, test_size=0.20)

## Took out TSI_chlA_mean, replaced with CHLAVESurfaceMean

#ignoredVars = ['Unnamed: 0', 'day', 'year', 'dates', 'scenario', 'month', 'reduction', 'abm']
ignoredVars = ['year', 'month', 'day', 'decade', 'season', 'abm', 'bmpAdp', 'bmpEff']
# ChlA Outputs
ignoredVars += ['CHLAVESurfaceMax','CHLAVESurfaceMean']
# TP Outputs
ignoredVars += ['TPSurfaceMax','TPSurfaceMean']
# Other Outputs
ignoredVars += ['SecchiSurfaceMax','SecchiSurfaceMean','TEMPSurfaceMax',
               'TEMPSurfaceMean']
# Ignored as Outputs by Pat, but not necessarily by Andrew -- Context Dependent
#ignoredVars += ['JPO4','JPOP','JPON','PO4T2R','POPR','PO4JRES']
# TP Columns that don't take redution scenarios into account
# ignoredVars += ['missisquoi_TP_PreReduction','pike_TP_PreReduction','rock_TP_PreReduction','basin_TP_gps_PreReduction']
# Deemed less important by Andrew
# ignoredVars += ['baseflow','X.sat_area','sat_def','trans']

x_train, y_train = df_train.drop(ignoredVars,axis=1), df_train['CHLAVESurfaceMean']
x_test, y_test = df_test.drop(ignoredVars,axis=1), df_test['CHLAVESurfaceMean']

x_train['random'] = np.random.random(size=len(x_train))
x_test['random'] = np.random.random(size=len(x_test))

print("Running Random Forest Regression...")
rf = RandomForestRegressor(n_estimators=100, n_jobs=12)
rf.fit(x_train, y_train)

print("Calculating Default Permutation Importances...")
imp = importances(rf, x_test, y_test) # default permutation algorithm

# Do we want to do this? -- Also takes a long while
print("Calculating Permutation Importances Using Cross Validation...")
cv_imp = cv_importances(rf, x_test, y_test, k=5) # permutation using cross validation

# Warning -- Takes forever! (5.5hr) on our 1.8 million rows
# print("Calculating Permutation Using Drop Column Importances...")
# dc_imp = dropcol_importances(rf, x_train, y_train, x_test, y_test) # permutation using drop column mechanism

# from eli5.sklearn import PermutationImportance
# perm = PermutationImportance(rf).fit(x_text, y_test)

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

viz = plot_importances(imp, title="Perm Import - Default")
viz.view()

cv_viz = plot_importances(cv_imp, title="Perm Import - Cross Val")
cv_viz.view()

# dc_viz = plot_importances(dc_imp, title="Perm Import - Drop Column")
# dc_viz.view()