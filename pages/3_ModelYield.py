import pandas as pd
import numpy as np
from scipy import interpolate
import streamlit as st

df_NormRef = pd.read_csv('Data/NormMS_Interp.csv')
df_agron = pd.read_csv('Data/GasExAgron.csv')

# List of spectral indices columns
indices_columns = ['Blue', 'Green', 'NIR', 'Red', 'RedEdge', 'SAVIMASK', 'NDVI', 
                   'GNDVI', 'RDVI', 'NLI', 'CVI', 'MSR', 'NDI', 'NDVIRedge', 
                   'PSRI', 'CIRedge', 'MTCI']

# Pivot the dataframe
wide_df = df_NormRef.pivot_table(
    index=['PLOT_YEAR', 'YEAR', 'GENOTYPE', 'PLOT', 'NTREATMENT'], 
    columns='DAP', 
    values=indices_columns
).reset_index()

# Flatten the multi-level column names
wide_df.columns = [
    f'{col[0]}_{col[1]}' if col[1] != '' else col[0] 
    for col in wide_df.columns
]

df_all = pd.merge(df_agron, wide_df, on=['PLOT_YEAR', 'YEAR', 'GENOTYPE', 'PLOT', 'NTREATMENT'])
df_all = df_all.dropna(axis=1)

st.markdown('If we flatten out the 16 Remote sensing spectra across 60 time points we have generated 16 * 60 = 960 features that can help predict yield')
st.dataframe(df_all)

y = df_all['KERNELDRYWT_PERPLANT']

s1 = list(df_all.columns[6:10])
s2 = list(df_all.columns[11:])
all_possible_features = s1 + s2
agron_features = all_possible_features[:6]
st.markdown('**All possible features**')
st.write(all_possible_features)

y = df_all['KERNELDRYWT_PERPLANT']

st.write(y)
x = df_all[all_possible_features]
st.write(x)
