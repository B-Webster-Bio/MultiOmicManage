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

st.dataframe(df_all)
y = df_all['KERNELDRYWT_PERPLANT']

s1 = list(df_all.columns[6:10])
s2 = list(df_all.columns[12:])
all_possible_features = s1 + s2
st.write(all_possible_features)