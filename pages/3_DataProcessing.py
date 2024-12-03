import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np

# point-cloud remote sensing 
df_pc = pd.read_csv('Raw\RGBAutoHts.csv')
# multispectral reflectance reflectance
df_ref = pd.read_csv('Raw\MultiSpec.csv')
# meta data about field plots
df_meta = pd.read_csv('Raw\MasterMetaData.csv')
df_meta.drop(columns=['FTREATMENT', 'TYPE', 'LOCATION', 'PLANTINGDATE'], inplace=True)
# hand collected agronomy data
df_agro = pd.read_csv('Raw\MasterAgronData.csv')
# gas excahnge data from Licor
df_gasex = pd.read_csv('Raw\MasterGasExData.csv')
df_gasex.drop(columns=['date', 'time'], inplace=True)

# formatting
dfs = [df_pc, df_ref, df_agro, df_meta, df_gasex]
for df in dfs:
    df['YEAR'] = df['YEAR'].astype('str')
    df['PLOT'] = df['PLOT'].astype('str')
    df['PLOT_YEAR'] = df['PLOT_YEAR'].astype('str')

# Subset those with gas_ex measurements
df_gasex = pd.merge(df_meta, df_gasex, on = ['PLOT_YEAR', 'PLOT', 'YEAR'], how='inner')

# before aggregating
plt.figure(figsize=(8, 5))
sns.heatmap(df_gasex.isna(), cmap="magma")
plt.title('Missing Before Aggregate')

# list of cols to calc mean
cols = list(df_gasex.columns[9:])
cols.append('PLOT_YEAR')
df_gasex_g = df_gasex[cols]
# calc mean based on unique PLOT_YEAR
df_gasex_g = df_gasex_g.groupby(by='PLOT_YEAR').mean().reset_index()

# Merge it back with meta data
df_gasex_g = pd.merge(df_meta, df_gasex_g, on = 'PLOT_YEAR', how='inner')
print('Total plot samples before: {}'.format(len(df_gasex_g.index)))

# plot after 
plt.figure(figsize=(8, 5))
sns.heatmap(df_gasex_g.isna(), cmap="magma")
plt.title('Missing After Aggregate')

# taking mean of plot_level resolved all missing except for one.
print(df_gasex_g.loc[df_gasex_g['A'].isna(),:])
# Not sure why this sample is missing let's look at yield and flowering

# before fill impution
# sort by year to see MAR
df_gasex_g.sort_values(by='YEAR', inplace=True)
plt.figure(figsize=(8, 5))
sns.heatmap(df_gasex_g.isna(), cmap="magma", vmin=0, vmax = 1)
plt.title('Missing Before Fill')

# impute missing
df_gasex_g['SUBPOPULATION'] = df_gasex_g.groupby('GENOTYPE')['SUBPOPULATION'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

# after 
plt.figure(figsize=(8, 5))
sns.heatmap(df_gasex_g.isna(), cmap="magma", vmin=0, vmax = 1)
plt.title('Missing After Fill')

# keys for merging
keys = ['PLOT_YEAR', 'PLOT', 'YEAR']

# merge agronomic traits of interest with gas ex
cols2keep = ['KERNELDRYWT_PERPLANT', 'KERNELMOISTURE_P', 'DAYSTOANTHESIS',
              'DAYSTOSILK', 'ASI', 'AVGFLAGHT_CM'] + keys

df_agro = df_agro.loc[:, cols2keep]

df_gasex_g = pd.merge(df_gasex_g, df_agro, on=keys, how='inner')

# all of the missing values are from 2023
df_gas_miss = df_gasex_g.loc[df_gasex_g['KERNELDRYWT_PERPLANT'].isna(), :]
print(df_gas_miss['YEAR'].value_counts())

df_gasex_g.sort_values(by='YEAR', inplace=True)
plt.figure(figsize=(8,5))
sns.heatmap(df_gasex_g.isna(), cmap="magma")

# Assess if the yield missingness is related to any other variable
df_gasex_g['yield_missing'] = df_gasex_g['KERNELDRYWT_PERPLANT'].isna().astype(int)
quantcols = list(df_gasex_g.columns[9:])
plt.figure(figsize=(12,8))
sns.heatmap(df_gasex_g.loc[df_gasex_g['YEAR'] == '2023', quantcols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()
quantcols.remove('yield_missing')
df_gasex_g.drop(columns = 'yield_missing', inplace = True)

# It looks like kernel weight might have outlier, let's scale and see how many sd 
numeric_cols = df_gasex_g.select_dtypes(include=[np.number]).columns
df_g_s = df_gasex_g[numeric_cols].apply(zscore)

# looks like one point is almost 5 SD above while everything else is > 3, let's remove
sns.scatterplot(data=df_g_s, x= df_g_s.index, y = 'KERNELDRYWT_PERPLANT')
plt.title('Yield ZScores Before')
plt.show()
df_gasex_g = df_gasex_g.loc[df_gasex_g['PLOT_YEAR'] != '7318_2023']

df_g_s = df_gasex_g[numeric_cols].apply(zscore)

# After
sns.scatterplot(data=df_g_s, x= df_g_s.index, y = 'KERNELDRYWT_PERPLANT')
plt.title('Yield ZScores After')
plt.show()

# Plot_Years of interest
plots2keep = set(df_gasex_g['PLOT_YEAR'].values)
# keys for merging
keys = ['PLOT_YEAR', 'PLOT', 'YEAR']
# select only those plots we are interested in
df_ref = df_ref.loc[df_ref['PLOT_YEAR'].isin(plots2keep)]

print(df_ref.isna().sum())
# The multispectral reflectance data is most useful when processed into vegetative indices 
# Let's make MS table now
# Prepare data
df_ref['ID'] = df_ref['PLOT_YEAR'].astype('str') + "_" + df_ref['DAP'].astype('str')
unmelt = df_ref.pivot(index = 'ID', columns = 'Band', values = 'MEAN').reset_index()
# Just doing this here to move it down later
unmelt['NDVI'] = (unmelt['NIR'] - unmelt['Red']) / (unmelt['NIR'] + unmelt['Red'])
unmelt['GNDVI'] = (unmelt['NIR'] - unmelt['Green']) / (unmelt['NIR'] + unmelt['Green'])
unmelt['RDVI'] = (unmelt['NIR'] - unmelt['Red']) / (np.sqrt(unmelt['NIR'] + unmelt['Red']))
unmelt['NLI'] = ((unmelt['NIR']**2) - unmelt['Red']) / ((unmelt['NIR']**2) + unmelt['Red'])
unmelt['CVI'] = (unmelt['NIR'] * unmelt['NIR']) / (unmelt['Green']**2)
unmelt['MSR'] = ((unmelt['NIR'] / unmelt['Red']) - 1) / ((np.sqrt(unmelt['NIR'] / unmelt['Red'])) + 1)
unmelt['NDI'] = (unmelt['RedEdge'] - unmelt['Red']) / (unmelt['RedEdge'] + unmelt['Red'])
unmelt['NDVIRedge'] = (unmelt['NIR'] - unmelt['RedEdge']) / (unmelt['NIR'] + unmelt['RedEdge'])
unmelt['PSRI'] = (unmelt['Red'] - unmelt['Blue']) / unmelt['RedEdge']
unmelt['CIRedge'] = (unmelt['NIR'] / unmelt['RedEdge']) - 1
unmelt['MTCI'] = (unmelt['NIR'] - unmelt['RedEdge']) / (unmelt['RedEdge'] - unmelt['Red'])

unmelt[['PLOT', 'YEAR', 'DAP']] = unmelt['ID'].str.split(pat = '_', n=2, expand = True)

df_ref = pd.merge(df_meta, unmelt, on =['PLOT', 'YEAR'], how = 'inner')
df_ref = df_ref.loc[df_ref['PLOT_YEAR'].isin(plots2keep), ]
df_ref['DAP'] = df_ref['DAP'].astype('int')

sns.lineplot(data=df_ref, x = 'DAP', y = 'NDVI', hue = 'YEAR')
plt.show()