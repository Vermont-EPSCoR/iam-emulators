### Gitlab site was great! - https://github.com/parrt/random-forest-importances
### rfpimp doesn't seem to be maintained anymore and doesn't work with current version of scikit-learn

### Re: Scaling Inputs for a RandomForestRegressor
###   - https://datascience.stackexchange.com/questions/74258/scikit-learn-random-forest-model-changes-as-result-of-input-scaling

### Use ciroh environment on epscor-pascal

import re
import os
import tempfile
from datetime import datetime
import pickle
from scenario_configuration import IAM_CONFIGURATION as settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay


threads_to_use = 20

# for ERL CSV, need 512G of Shared Memory
# sudo mount -o remount,size=512G /dev/shm

# Turn on pandas copy on write optimizations
pd.set_option("mode.copy_on_write", True)

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

def concentration_to_class(conc):
	if conc < 2.6:
		return 0
	elif conc < 10:
		return 1
	elif conc < 40:
		return 2
	else:
		return 3


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

### Commenting out because rfpimp hasn't been updated since 2021
###   and currently doesn't work (asks for an old sklearn import)
###   https://github.com/parrt/random-forest-importances
# def do_importances_V1(rf, x_test, y_test):
# 	from rfpimp import importances, cv_importances, plot_importances

# 	print("Calculating Default Permutation Importances...")
# 	imp = importances(rf, x_test, y_test) # default permutation algorithm

# 	# Do we want to do this? -- Also takes a long while
# 	print("Calculating Permutation Importances Using Cross Validation...")
# 	cv_imp = cv_importances(rf, x_test, y_test, k=5) # permutation using cross validation

# 	# Warning -- Takes forever! (5.5hr) on our 1.8 million rows
# 	# print("Calculating Permutation Using Drop Column Importances...")
# 	# dc_imp = dropcol_importances(rf, x_train, y_train, x_test, y_test) # permutation using drop column mechanism

# 	# from eli5.sklearn import PermutationImportance
# 	# perm = PermutationImportance(rf).fit(x_text, y_test)

# 	print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 	viz = plot_importances(imp, title="Perm Import - Default")
# 	viz.view()

# 	cv_viz = plot_importances(cv_imp, title="Perm Import - Cross Val")
# 	cv_viz.view()

# 	# dc_viz = plot_importances(dc_imp, title="Perm Import - Drop Column")
# 	# dc_viz.view()

def do_importances(rf, x_test, y_test, csv):
	from sklearn.inspection import permutation_importance

	### This process needs a bunch of disk space to create the permutations...
	### Can probably take this out... it was /dev/shm that was running out of space
	# os.environ['TMPDIR'] = '/raid/tmp'

	# with tempfile.TemporaryDirectory() as path:
	# 	os.chdir('/raid/tmp')
	print("Running permutations...")
	imp = permutation_importance(
		rf,
		np.array(x_test.to_numpy(), copy=True),
		np.array(y_test.to_numpy(), copy=True),
		n_repeats=10,
		n_jobs=threads_to_use)
	
	with open("importance.pickle", 'wb') as file_out:
		pickle.dump(imp, file_out)
	
	print("Creating results DataFrame...")
	imp_df = pd.DataFrame(
		{
			"mean": imp.importances_mean,
			"std": imp.importances_std
		},
		index = x_test.columns
		).sort_values(by=['mean'], ascending=True)
	
	print("Plotting results...")
	import matplotlib.pyplot as plt 

	fig, ax = plt.subplots(figsize=(10,16))
	imp_df['mean'].plot.barh(yerr=imp_df['std'], ax=ax)
	plt.suptitle("SKLearn feature importances\n using permutation on full model")
	ax.set_title(csv, fontsize="small")
	ax.set_ylabel("Mean accuracy decrease")
	fig.tight_layout()
	plt.show()

def do_importances_mdi(rf, x_test, y_test, csv):
	
	imp_mean = rf.feature_importances_
	imp_std = np.std([rf.feature_importances_ for tree in rf.estimators_], axis=0)
	
	imp_df = pd.DataFrame(
		{
			"mean": imp_mean,
			"std": imp_std
		},
		index = x_test.columns
		).sort_values(by=['mean'], ascending=True)	
	
	fig, ax = plt.subplots(figsize=(10,16))
	imp_df['mean'].plot.barh(yerr=imp_df['std'], ax=ax)
	plt.suptitle("SKLearn feature importances\n using mean decrease in impurity (MDI)")
	ax.set_title(csv.filename, fontsize="small")
	ax.set_ylabel("Mean decrease in impurity")
	fig.tight_layout()
	# plt.show()
	plt.savefig(f'Graph{csv.name}MDIImportance.png')

def partial_dependence_plot(rf, x_train, csv):
	variables = ['JPO4', 'POPR', 'JPON', 'sat_def', 'snowpack', 'PO4T2R', 'JPOP', 'baseflow', r'%sat_area', '0_redux', 'evap', 'trans']

	fig = plt.figure(figsize=(10,16))
	plt.rcParams['font.size'] = 8
	plt.suptitle(f'Partial Dependence Plots for {csv.name}')
	pdd = PartialDependenceDisplay.from_estimator(rf, x_train, variables, ax=plt.gca(), n_jobs=20, verbose=1)
	plt.subplots_adjust(top=0.95)
	# plt.show()
	plt.savefig(f'Graph{csv.name}PDPs.png')

	print("Creating individual PDPs...")
	for variable in variables:
		print(f"   {variable}")
		pdd = PartialDependenceDisplay.from_estimator(rf, x_train, [variable], n_jobs=20, verbose=1)
		plt.gca().set_title(f'Partial Dependence Plot for {variable}')
		plt.savefig(f'PDP_{variable}.png')
	
	# print("Saving figure...")
	# with open(csv.name+"_figure.pickle", 'wb') as out_file:
	# 	pickle.dump(pdd, out_file)
	
	# fig = plt.figure(figsize=(10,16))
	# plt.suptitle(f'Partial Dependence Plots for {csv.name}', y=.9)
	# pdd.plot(ax=plt.gca())
	# plt.show()

def partial_dependence_plot_classifier(rf, x_train, csv):
	variables = ['JPO4', 'POPR', 'JPON', 'sat_def', 'snowpack', 'PO4T2R', 'JPOP', 'baseflow', r'%sat_area', '0_redux', 'evap', 'trans']

	for CHLAclass in range(0,4):
		print(f"Class: {CHLAclass}")
		fig = plt.figure(figsize=(10,16))
		plt.rcParams['font.size'] = 8
		plt.suptitle(f'Partial Dependence Plots for {csv.name}: Class {CHLAclass}')
		pdd = PartialDependenceDisplay.from_estimator(rf, x_train, variables, target=CHLAclass, ax=plt.gca(), n_jobs=20, verbose=1)
		plt.subplots_adjust(top=0.95)
		# plt.show()
		plt.savefig(f'Graph{csv.name}PDPsClass{CHLAclass}.png')

		print("Creating individual PDPs...")
		for variable in variables:
			print(f"   {variable}: Class {CHLAclass}")
			pdd = PartialDependenceDisplay.from_estimator(rf, x_train, [variable], target=CHLAclass, n_jobs=20, verbose=1)
			plt.gca().set_title(f'Partial Dependence Plot for {variable}: Class {CHLAclass}')
			plt.savefig(f'PDP_Class{CHLAclass}_{variable}.png')

	
	# print("Saving figure...")
	# with open(csv.name+"_figure.pickle", 'wb') as out_file:
	# 	pickle.dump(pdd, out_file)
	
	# fig = plt.figure(figsize=(10,16))
	# plt.suptitle(f'Partial Dependence Plots for {csv.name}', y=.9)
	# pdd.plot(ax=plt.gca())
	# plt.show()


class DataCSV():
	ignored_vars = ['year', 'month', 'day', 'decade', 'season', 'abm']
	# ChlA Outputs
	ignored_vars += ['CHLAVESurfaceMax']
	# TP Outputs
	ignored_vars += ['TPSurfaceMax', 'TPSurfaceMean']
	# Other Outputs
	ignored_vars += ['SecchiSurfaceMax', 'SecchiSurfaceMean', 'TEMPSurfaceMax', 'TEMPSurfaceMean']
	# Add CHLAVE Mean to use class variable instead
	ignored_vars += ['CHLAVESurfaceMean']

	### Ignored as Outputs by Pat, but not necessarily by Andrew -- Context Dependent
	# ignoredVars += ['JPO4','JPOP','JPON','PO4T2R','POPR','PO4JRES']
	### Deemed less important by Andrew
	# ignoredVars += ['baseflow','X.sat_area','sat_def','trans']
	### TP Columns that don't take redution scenarios into account
	# ignoredVars += ['missisquoi_TP_PreReduction','pike_TP_PreReduction','rock_TP_PreReduction','basin_TP_gps_PreReduction']

	def get_ignored_vars(self):
		return self.ignored_vars
	
	def __init__(self, filename, name):
		self.filename = filename
		self.name = name

		
class ERLCSV(DataCSV):
	def get_ignored_vars(cls):
		return super().get_ignored_vars() + ['bmpAdp', 'bmpEff']


class ComboCSV(DataCSV):
	def get_ignored_vars(cls):
		return super().get_ignored_vars() + ['IAMRun']


def load_dataset(csv):
	print("Loading file...")
	# Use nrows parameter to limit size (104000 = 2031)
	# df = pd.read_csv('/raid/pclemins/random-forest/sediment_tabular_data_apr-nov_until_2050_includes_TSIchl_with_inputs_TP_Reductions_LandUse.csv')
	# df = pd.read_csv('/raid/pclemins/IAMData-P_Reduction_V2/sediment_tabular_data.csv')
	df = pd.read_csv(csv.filename)

	print(f'Imported csv size: {df.shape}')

	print('Filtering to include only Apr - Nov...')
	df = df[(df['month'] >= 4) & (df['month'] <= 11)]
	print(f'Filtered csv size: {df.shape}')

	print('Filtering out data after 2047...')
	df = df[(df['year'] <= 2047)]
	print(f'Filtered csv size: {df.shape}')

	### Commenting out landuse_proportion for now...
	# print("Loading landuse_proportion files...")
	# landuseDF = pd.DataFrame()
	# for file in glob.glob('/epscorfs/iam-workspace/pclemins/workflow/P_Reduction_V2/outputs/*landuse_proportions.R'):
	#   result = re.search(scenario_regex,os.path.basename(file)).groupdict()
	#   scenario = result['gcm']+'.'+result['rcp']+'.'+result['p_redux']+'.'+result['abm']
	#   with open(file, "r") as openFile:
	#     for line in openFile:
	#       lineparts = line.split("=")
	#       if len(lineparts) == 2:
	#         landuseDF.loc[result['decade']+'.'+scenario, lineparts[0].strip()] = lineparts[1].strip()

	# print("Adding abm landuse proportion columns...")
	# for luType in ['undev','dev','ag']:
	#   print("  "+luType+"...")
	#   df['abm-'+luType] = df.apply(lambda row: landuseDF.loc[str(
	#     row['decade']) + '.' +
	#     row['gcm'] + '.' +
	#     row['rcp'] + '.' +
	#     row['reduxP'] + '.' +
	#     row['abm'],
	#     'proportion_'+luType], axis=1)

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

	print("Adding CHLA class variable...")
	df['CHLAClass'] = df['CHLAVESurfaceMean'].apply(concentration_to_class)
	
	df = df.drop(csv.get_ignored_vars(), axis=1)

	print("Replacing Strings with Multiple 0/1 Columns...")
	for stringVar in ['gcm', 'rcp', 'reduxP']:
		df = pd.concat([df, StringToMultipleBinary(df[stringVar])], axis=1)
		df = df.drop(stringVar, axis=1)
	
	if csv.name == "ERL":
		print('Adding random variable...')
		df['RandomNum'] = np.random.random(size=len(df))
	
	return df


#########
# Main Code
#########

fit_model = False
create_train_test = False

csv = ComboCSV('/raid/pclemins/IAMData-comboCSV/comboCSV.csv', 'comboCSV')
#csv = ERLCSV('/raid/pclemins/IAMData-P_Reduction_V2/sediment_tabular_data.csv', 'ERL')

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if create_train_test:
	df = load_dataset(csv)

	print("Analysis DataFrame Columns:")
	print(df.columns)
	df[df.isnull().any(axis=1)].to_csv(csv.name+'_nan.csv')

	df_train, df_test = train_test_split(df, test_size=0.20)
	df_train.to_csv(csv.name+'_train.csv', index=False)
	df_test.to_csv(csv.name+'_test.csv', index=False)

	df_train.iloc[np.random.choice(len(df_train), size=1000, replace=False)].to_csv(csv.name+'_train_1000.csv', index=False)
	df_train.iloc[np.random.choice(len(df_train), size=5000, replace=False)].to_csv(csv.name+'_train_5000.csv', index=False)

else:
	print('Loading training and testing datasets...')
	df_train = pd.read_csv(csv.name+'_train.csv')
	df_test = pd.read_csv(csv.name+'_test.csv')

## Took out TSI_chlA_mean, replaced with CHLAVESurfaceMean
# target_variable = 'CHLAVESurfaceMean'
target_variable = 'CHLAClass'

x_train, y_train = df_train.drop(target_variable, axis=1), df_train[target_variable]
x_test, y_test = df_test.drop(target_variable, axis=1), df_test[target_variable]

if fit_model:
	print("Training random forest...")
	# rf = RandomForestRegressor(n_estimators=100, n_jobs=threads_to_use, verbose=1)
	rf = RandomForestClassifier(n_estimators=100, n_jobs=threads_to_use, verbose=1)
	rf.fit(x_train, y_train)

	print("Saving trained model...")
	with open(csv.name+".pickle", 'wb') as out_file:
		pickle.dump(rf, out_file)

else:
	print("Loading trained model...")
	with open(csv.name+".pickle", 'rb') as in_file:
		rf = pickle.load(in_file)

print("Creating Test Train Eval plot...")
eval = rf.predict(x_test)

hists = [(y_train, "Training"),
		 (y_test, "Testing Actual"),
		 (eval, "Testing Model")]
fig, axes = plt.subplots(nrows=1, ncols=3)
for axis, plot in zip(axes.flatten(), hists):
	axis.hist(plot[0])
	axis.set_title(plot[1])
plt.savefig(f'Graph{csv.name}TrainTestEval.png')

print("Running importance analysis...")
do_importances_mdi(rf, x_test, y_test, csv)

print("Creating Partial Dependence plots...")
# 500 is about memory max for regressor, but need more for classifier
size = 1000 
idx = np.random.choice(len(x_train), size=500, replace=False)
partial_dependence_plot_classifier(rf, x_train.iloc[idx], csv)
