from scenario_configuration import IAM_CONFIGURATION as settings
import pandas as pd

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


print("Loading file...")
# Use nrows parameter to limit size (104000 = 2031)
df = pd.read_csv('/raid/pclemins/random-forest/sediment_tabular_data_apr-nov_until_2050_icludes_TSIchl_with_inputs.csv')

print(f'Imported csv of size {df.shape}')

print("Calculating phosphorus reduction for each row...")
df['reduction'] = df.apply(lambda row: DetermineReduction(row['redux'], row['year']), axis=1)
for TPcolumn in ['missisquoi_TP','pike_TP','rock_TP','basin_TP_gps']:
  print(f"Applying phosphorus reduction to {TPcolumn}...")
  df[f'{TPcolumn}_PreReduction'] = df[TPcolumn]
  df[TPcolumn] = df[TPcolumn] * ((100-df['reduction'])/100)

print("Writing file...")
df.to_csv(path_or_buf='/raid/pclemins/random-forest/sediment_tabular_data_apr-nov_until_2050_includes_TSIchl_with_inputs_TP_Reductions.csv', index=False)
