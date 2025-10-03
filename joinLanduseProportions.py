import re
import glob
import os
import pandas as pd
import math

scenario_regex = re.compile(''.join([
        r'^(?:(?P<decade>[0-9]{4})-)?',
        r'(?P<abm>.+?)-',
        r'(?P<gcm>.+?)-',
        r'(?P<rcp>(?:rcp)?[0-9]{1,3}|NORCP)',
        r'(?:-(?P<p_redux>[A-Za-z0-9]+_redux))?',
        r'-landuse_proportions\.R'
    ]))

print("Loading landuse_proportion files...")
landuseDF = pd.DataFrame()
for file in glob.glob('/epscorfs/iam-workspace/pclemins/workflow/P_Reduction_V2/outputs/*landuse_proportions.R'):
  result = re.search(scenario_regex,os.path.basename(file)).groupdict()
  scenario = result['gcm']+'.'+result['rcp']+'.'+result['p_redux']+'.'+result['abm']
  thisDict = dict()
  with open(file, "r") as openFile:
    for line in openFile:
      lineparts = line.split("=")
      if len(lineparts) == 2:
        landuseDF.loc[result['decade']+'.'+scenario, lineparts[0].strip()] = lineparts[1].strip()
  
print("Loading big csv...")
# Use nrows parameter to limit size (104000 = 2031)
df = pd.read_csv('/raid/pclemins/random-forest/sediment_tabular_data_apr-nov_until_2050_includes_TSIchl_with_inputs_TP_Reductions.csv')

print("Adding abm landuse proportion columns...")
for luType in ['undev','dev','ag']:
  print("  "+luType+"...")
  df['decade'] = (df['year']-1) // 10 * 10 + 1
  df['abm-'+luType] = df.apply(lambda row: landuseDF.loc[str(row['decade'])+'.'+row['scenario'], 'proportion_'+luType], axis=1)

df = df.drop('decade', axis=1)

print("Writing file...")
df.to_csv(path_or_buf='/raid/pclemins/random-forest/sediment_tabular_data_apr-nov_until_2050_includes_TSIchl_with_inputs_TP_Reductions_LandUse.csv', index=False)
