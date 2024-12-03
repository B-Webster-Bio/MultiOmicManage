import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np

def load_data():
    """Load all necessary data files"""
    # Multispectral reflectance 
    df_ref = pd.read_csv('Raw/MultiSpec.csv')
    # Meta data about field plots
    df_meta = pd.read_csv('Raw/MasterMetaData.csv')
    df_meta.drop(columns=['FTREATMENT', 'TYPE', 'LOCATION', 'PLANTINGDATE'], inplace=True)
    # Hand collected agronomy data
    df_agro = pd.read_csv('Raw/MasterAgronData.csv')
    # Gas exchange data from Licor
    df_gasex = pd.read_csv('Raw/MasterGasExData.csv')
    df_gasex.drop(columns=['date', 'time'], inplace=True)

    return df_ref, df_meta, df_agro, df_gasex

def preprocess_data(df_ref, df_meta, df_agro, df_gasex):
    """Preprocess and merge datasets"""
    # Formatting
    dfs = [df_ref, df_agro, df_meta, df_gasex]
    for df in dfs:
        df['YEAR'] = df['YEAR'].astype('str')
        df['PLOT'] = df['PLOT'].astype('str')
        df['PLOT_YEAR'] = df['PLOT_YEAR'].astype('str')

    # Subset those with gas exchange measurements
    df_gasex = pd.merge(df_meta, df_gasex, on=['PLOT_YEAR', 'PLOT', 'YEAR'], how='inner')

    # Aggregate gas exchange data
    cols = list(df_gasex.columns[9:])
    cols.append('PLOT_YEAR')
    df_gasex_g = df_gasex[cols]
    df_gasex_g = df_gasex_g.groupby(by='PLOT_YEAR').mean().reset_index()

    # Merge back with meta data
    df_gasex_g = pd.merge(df_meta, df_gasex_g, on='PLOT_YEAR', how='inner')

    # Impute missing values
    df_gasex_g['SUBPOPULATION'] = df_gasex_g.groupby('GENOTYPE')['SUBPOPULATION'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

    # Merge with agronomic traits
    keys = ['PLOT_YEAR', 'PLOT', 'YEAR']
    cols2keep = ['KERNELDRYWT_PERPLANT', 'KERNELMOISTURE_P', 'DAYSTOANTHESIS',
                 'DAYSTOSILK', 'ASI', 'AVGFLAGHT_CM'] + keys
    df_agro = df_agro.loc[:, cols2keep]
    df_gasex_g = pd.merge(df_gasex_g, df_agro, on=keys, how='inner')

    # Remove outliers
    df_gasex_g = df_gasex_g.loc[df_gasex_g['PLOT_YEAR'] != '7318_2023']

    # Process multispectral reflectance data
    plots2keep = set(df_gasex_g['PLOT_YEAR'].values)
    df_ref = df_ref.loc[df_ref['PLOT_YEAR'].isin(plots2keep)]

    # Calculate vegetative indices
    df_ref['ID'] = df_ref['PLOT_YEAR'].astype('str') + "_" + df_ref['DAP'].astype('str')
    unmelt = df_ref.pivot(index='ID', columns='Band', values='MEAN').reset_index()
    
    # Calculate various indices
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

    unmelt[['PLOT', 'YEAR', 'DAP']] = unmelt['ID'].str.split(pat='_', n=2, expand=True)

    df_ref = pd.merge(df_meta, unmelt, on=['PLOT', 'YEAR'], how='inner')
    df_ref = df_ref.loc[df_ref['PLOT_YEAR'].isin(plots2keep), ]
    df_ref['DAP'] = df_ref['DAP'].astype('int')

    return df_gasex_g, df_ref

def main():
    st.title('Multispectral Data Analysis')

    # Load data
    st.sidebar.header('Data Loading')
    if st.sidebar.button('Load Data'):
        with st.spinner('Loading and preprocessing data...'):
            try:
                df_ref, df_meta, df_agro, df_gasex = load_data()
                df_gasex_g, df_ref = preprocess_data(df_ref, df_meta, df_agro, df_gasex)
                st.session_state.df_gasex_g = df_gasex_g
                st.session_state.df_ref = df_ref
                st.success('Data loaded successfully!')
            except Exception as e:
                st.error(f'Error loading data: {e}')

    # Data Visualization
    st.sidebar.header('Visualizations')
    
    # Check if data is loaded
    if 'df_gasex_g' not in st.session_state or 'df_ref' not in st.session_state:
        st.warning('Please load data first using the sidebar.')
        return

    df_gasex_g = st.session_state.df_gasex_g
    df_ref = st.session_state.df_ref

    # Missing Data Heatmap
    if st.sidebar.checkbox('Show Missing Data Heatmap'):
        st.subheader('Missing Data Heatmap')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_gasex_g.isna(), cmap="magma", ax=ax)
        st.pyplot(fig)

    # Outlier Detection (Z-Score Plot)
    if st.sidebar.checkbox('Show Yield Z-Score Plot'):
        st.subheader('Yield Z-Score Plot')
        numeric_cols = df_gasex_g.select_dtypes(include=[np.number]).columns
        df_g_s = df_gasex_g[numeric_cols].apply(zscore)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_g_s, x=df_g_s.index, y='KERNELDRYWT_PERPLANT', ax=ax)
        ax.set_title('Yield ZScores')
        st.pyplot(fig)

    # NDVI Line Plot
    if st.sidebar.checkbox('Show NDVI Over Time'):
        st.subheader('NDVI by Year')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df_ref, x='DAP', y='NDVI', hue='YEAR', ax=ax)
        st.pyplot(fig)

    # Correlation Heatmap
    if st.sidebar.checkbox('Show Correlation Heatmap'):
        st.subheader('Correlation Heatmap')
        quantcols = list(df_gasex_g.select_dtypes(include=[np.number]).columns)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_gasex_g[quantcols].corr(), annot=True, cmap='coolwarm', 
                    vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)

if __name__ == '__main__':
    main()