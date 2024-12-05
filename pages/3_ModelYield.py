import pandas as pd
import numpy as np
from scipy import interpolate
import streamlit as st

df_NormRef = pd.read_csv('Data/NormMS.csv')


def interpolate_spectral_indices(df):
    # Group by PLOT_YEAR and GENOTYPE
    grouped = df.groupby(['PLOT_YEAR', 'GENOTYPE'])
    
    # Initialize list to store interpolated dataframes
    interpolated_dfs = []
    
    # Common DAP range to interpolate to
    common_dap_range = range(55, 99)
    
    for (plot_year, genotype), group in grouped:
        # Create a dataframe with the common DAP range
        interpolated_df = pd.DataFrame({'DAP': common_dap_range})
        
        # Interpolation for each spectral index
        indices_columns = ['Blue', 'Green', 'NIR', 'Red', 'RedEdge', 'SAVIMASK', 
                           'NDVI', 'GNDVI', 'RDVI', 'NLI', 'CVI', 'MSR', 
                           'NDI', 'NDVIRedge', 'PSRI', 'CIRedge', 'MTCI']
        
        for index in indices_columns:
            # Sort original data by DAP
            sorted_data = group.sort_values('DAP')
            
            # Linear interpolation
            f = interpolate.interp1d(sorted_data['DAP'], sorted_data[index], 
                                     kind='linear', fill_value='extrapolate')
            
            # Interpolate values for common DAP range
            interpolated_values = f(common_dap_range)
            interpolated_df[index] = interpolated_values
        
        # Add metadata columns
        interpolated_df['PLOT_YEAR'] = plot_year
        interpolated_df['GENOTYPE'] = genotype
        interpolated_df['YEAR'] = group['YEAR'].iloc[0]
        interpolated_df['PLOT'] = group['PLOT'].iloc[0]
        interpolated_df['NTREATMENT'] = group['NTREATMENT'].iloc[0]
        
        interpolated_dfs.append(interpolated_df)
    
    # Combine all interpolated dataframes
    final_interpolated_df = pd.concat(interpolated_dfs, ignore_index=True)
    
    return final_interpolated_df


# Perform interpolation
interpolated_df = interpolate_spectral_indices(df_NormRef)

st.dataframe(interpolated_df)


'''

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
'''