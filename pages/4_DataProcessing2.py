import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np

# Streamlit app title
st.title("Data Processing and Visualization")

# Load datasets
st.header("Load Data")
try:
    df_ref = pd.read_csv('Raw/MultiSpec.csv')
    df_meta = pd.read_csv('Raw/MasterMetaData.csv')
    df_agro = pd.read_csv('Raw/MasterAgronData.csv')
    df_gasex = pd.read_csv('Raw/MasterGasExData.csv')
    st.success("Datasets loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Data preprocessing
df_meta.drop(columns=['FTREATMENT', 'TYPE', 'LOCATION', 'PLANTINGDATE'], inplace=True)
df_gasex.drop(columns=['date', 'time'], inplace=True)

# Formatting
dfs = [df_ref, df_agro, df_meta, df_gasex]
for df in dfs:
    df['YEAR'] = df['YEAR'].astype('str')
    df['PLOT'] = df['PLOT'].astype('str')
    df['PLOT_YEAR'] = df['PLOT_YEAR'].astype('str')

# Merge
df_gasex = pd.merge(df_meta, df_gasex, on=['PLOT_YEAR', 'PLOT', 'YEAR'], how='inner')

st.subheader('Gas Exchange Data:')
st.markdown('Gas Exchange is collected twice per plot so we will take average on the plot level.')
# Visualize missing data
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df_gasex.isna(), cmap="magma", ax=ax)
plt.title("Missing Data Before Mean Aggregation")
st.pyplot(fig)

# Aggregation and merging
cols = list(df_gasex.columns[9:]) + ['PLOT_YEAR']
df_gasex_g = df_gasex[cols].groupby(by='PLOT_YEAR').mean().reset_index()
df_gasex_g = pd.merge(df_meta, df_gasex_g, on='PLOT_YEAR', how='inner')

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df_gasex_g.isna(), cmap="magma", ax=ax)
plt.title('Missing After Aggregate')
st.pyplot(fig)

# Imputation
st.subheader('Missing Meta Data')
st.markdown('Meta data from one year was not listed but we can use the fwd fill method to fill in missing entries with similar ones.')
df_gasex_g.sort_values(by='YEAR', inplace=True)
df_gasex_g['SUBPOPULATION'] = df_gasex_g.groupby('GENOTYPE')['SUBPOPULATION'].transform(
    lambda x: x.fillna(method='ffill').fillna(method='bfill'))

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df_gasex_g.isna(), cmap="magma", ax=ax)
plt.title('Missing After Fill')
st.pyplot(fig)


# keys for merging
keys = ['PLOT_YEAR', 'PLOT', 'YEAR']
# merge agronomic traits of interest with gas ex
cols2keep = ['KERNELDRYWT_PERPLANT',
              'DAYSTOSILK', 'AVGFLAGHT_CM'] + keys
df_agro = df_agro.loc[:, cols2keep]
df_gasex_g = pd.merge(df_gasex_g, df_agro, on=keys, how='inner')

# Assess if the yield missingness is related to any other variable
df_gasex_g['yield_missing'] = df_gasex_g['KERNELDRYWT_PERPLANT'].isna().astype(int)
quantcols = list(df_gasex_g.columns[9:])
st.header("Missingness correlation with trait values")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df_gasex_g.loc[df_gasex_g['YEAR'] == '2023', quantcols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1,
            ax=ax)
st.pyplot(fig)
quantcols.remove('yield_missing')
df_gasex_g.drop(columns = 'yield_missing', inplace = True)

cols2keep = quantcols
cols2keep.append('YEAR')
df_gasex_g = df_gasex_g.dropna()

# It looks like kernel weight might have outlier, let's scale and see how many sd 
numeric_cols = ['KERNELDRYWT_PERPLANT', 'DAYSTOSILK', 'AVGFLAGHT_CM']
df_g_s = df_gasex_g[numeric_cols].apply(zscore)

# looks like one point is almost 5 SD above while everything else is > 3, let's remove
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df_g_s, x= df_g_s.index, y = 'KERNELDRYWT_PERPLANT', ax=ax)
plt.title('Yield ZScores Before')
ax.hlines(y=3, linestyles='dashed')
st.pyplot(fig)

df_gasex_g = df_gasex_g.loc[df_gasex_g['PLOT_YEAR'] != '7318_2023']
df_g_s = df_gasex_g[numeric_cols].apply(zscore)

# After
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df_g_s, x= df_g_s.index, y = 'KERNELDRYWT_PERPLANT', ax = ax)
plt.title('Yield ZScores After')
st.pyplot(fig)

# Plot_Years of interest
plots2keep = set(df_gasex_g['PLOT_YEAR'].values)
# keys for merging
keys = ['PLOT_YEAR', 'PLOT', 'YEAR']
# select only those plots we are interested in
df_ref = df_ref.loc[df_ref['PLOT_YEAR'].isin(plots2keep)]


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

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=df_ref, x = 'DAP', y = 'NDVI', hue = 'YEAR', ax=ax)
plt.show()