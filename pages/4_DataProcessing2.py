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

# Visualize missing data
st.header("Missing Data Before Aggregation")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df_gasex.isna(), cmap="magma", ax=ax)
plt.title('Missing Before Aggregate')
st.pyplot(fig)

# Aggregation and merging
cols = list(df_gasex.columns[9:]) + ['PLOT_YEAR']
df_gasex_g = df_gasex[cols].groupby(by='PLOT_YEAR').mean().reset_index()
df_gasex_g = pd.merge(df_meta, df_gasex_g, on='PLOT_YEAR', how='inner')

st.header("Missing Data After Aggregation")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df_gasex_g.isna(), cmap="magma", ax=ax)
plt.title('Missing After Aggregate')
st.pyplot(fig)

# Imputation
df_gasex_g.sort_values(by='YEAR', inplace=True)
df_gasex_g['SUBPOPULATION'] = df_gasex_g.groupby('GENOTYPE')['SUBPOPULATION'].transform(
    lambda x: x.fillna(method='ffill').fillna(method='bfill'))

st.header("Missing Data After Imputation")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df_gasex_g.isna(), cmap="magma", ax=ax)
plt.title('Missing After Fill')
st.pyplot(fig)

# keys for merging
keys = ['PLOT_YEAR', 'PLOT', 'YEAR']

# merge agronomic traits of interest with gas ex
cols2keep = ['KERNELDRYWT_PERPLANT', 'KERNELMOISTURE_P', 'DAYSTOANTHESIS',
              'DAYSTOSILK', 'ASI', 'AVGFLAGHT_CM'] + keys

df_agro = df_agro.loc[:, cols2keep]

df_gasex_g = pd.merge(df_gasex_g, df_agro, on=keys, how='inner')

# Outlier detection and removal
numeric_cols = df_gasex_g.select_dtypes(include=[np.number]).columns
df_g_s = df_gasex_g[numeric_cols].apply(zscore)

st.header("Yield Z-Scores Before Outlier Removal")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df_g_s, x=df_g_s.index, y='KERNELDRYWT_PERPLANT', ax=ax)
plt.title('Yield ZScores Before')
st.pyplot(fig)

df_gasex_g = df_gasex_g.loc[df_gasex_g['PLOT_YEAR'] != '7318_2023']
df_g_s = df_gasex_g[numeric_cols].apply(zscore)

st.header("Yield Z-Scores After Outlier Removal")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df_g_s, x=df_g_s.index, y='KERNELDRYWT_PERPLANT', ax=ax)
plt.title('Yield ZScores After')
st.pyplot(fig)

# Vegetative indices calculation
df_ref = df_ref.loc[df_ref['PLOT_YEAR'].isin(df_gasex_g['PLOT_YEAR'])]
df_ref['ID'] = df_ref['PLOT_YEAR'] + "_" + df_ref['DAP'].astype('str')
unmelt = df_ref.pivot(index='ID', columns='Band', values='MEAN').reset_index()
unmelt['NDVI'] = (unmelt['NIR'] - unmelt['Red']) / (unmelt['NIR'] + unmelt['Red'])
unmelt['GNDVI'] = (unmelt['NIR'] - unmelt['Green']) / (unmelt['NIR'] + unmelt['Green'])
unmelt['RDVI'] = (unmelt['NIR'] - unmelt['Red']) / (np.sqrt(unmelt['NIR'] + unmelt['Red']))
unmelt[['PLOT', 'YEAR', 'DAP']] = unmelt['ID'].str.split('_', expand=True)

df_ref = pd.merge(df_meta, unmelt, on=['PLOT', 'YEAR'], how='inner')
df_ref['DAP'] = df_ref['DAP'].astype('int')

st.header("NDVI Over Time")
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=df_ref, x='DAP', y='NDVI', hue='YEAR', ax=ax)
plt.title('NDVI Across Time')
st.pyplot(fig)

st.success("Data processing and visualization complete!")