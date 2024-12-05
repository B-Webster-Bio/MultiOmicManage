import pandas as pd
import numpy as np
from scipy import interpolate
import streamlit as st

df_NormRef = pd.read_csv('Data/NormMS_Interp.csv')

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

st.dataframe(wide_df)