import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy import interpolate
import numpy as np

# Streamlit app title
st.title("Data Processing")
st.markdown('''All the data is currently stored as raw data and meta data with "Plot_Year" serving as a key 
            to match values. On this page you can see how merging the meta data with raw data, handling missing values, 
            removing outliers, and processing of remote sensing data was done.''')

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

with st.expander('Handle Missing Data and Cleaning'):
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
    st.subheader("Missingness correlation with trait values")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_gasex_g.loc[df_gasex_g['YEAR'] == '2023', quantcols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1,
            ax=ax)
    st.pyplot(fig)
    quantcols.remove('yield_missing')
    df_gasex_g.drop(columns = 'yield_missing', inplace = True)

    df_gasex_g.sort_values(by='YEAR', inplace=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_gasex_g.isna(), cmap="magma", ax = ax)
    st.pyplot(fig)
    st.markdown('All the missing yield is from 2023 and does not appear to be related to other traits so let us drop missing')
    cols2keep = quantcols
    cols2keep.append('YEAR')
    df_gasex_g = df_gasex_g.dropna()


# It looks like kernel weight might have outlier, let's scale and see how many sd 
numeric_cols = ['KERNELDRYWT_PERPLANT', 'DAYSTOSILK', 'AVGFLAGHT_CM']
df_g_s = df_gasex_g[numeric_cols].apply(zscore)

with st.expander('Outlier Detection with Z-Score'):

    # looks like one point is almost 5 SD above while everything else is > 3, let's remove
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_g_s, x= df_g_s.index, y = 'KERNELDRYWT_PERPLANT', ax=ax)
    plt.title('Yield ZScores Before')
    st.pyplot(fig)
    st.markdown('Remove the point with a z-score about 5')
    df_gasex_g = df_gasex_g.loc[df_gasex_g['PLOT_YEAR'] != '7318_2023']
    df_g_s = df_gasex_g[numeric_cols].apply(zscore)

    # After
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_g_s, x= df_g_s.index, y = 'KERNELDRYWT_PERPLANT', ax = ax)
    plt.title('Yield ZScores After')
    st.pyplot(fig)

    #Silking
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_g_s, x= df_g_s.index, y = 'DAYSTOSILK', ax=ax)
    plt.title('Silking ZScores')
    st.pyplot(fig)
    st.markdown('Looks good, you can see difference between years but that is not rare')

    # Height
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_g_s, x= df_g_s.index, y = 'AVGFLAGHT_CM', ax=ax)
    plt.title('Height ZScores')
    st.pyplot(fig)
    st.markdown('Looks good!')

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

with st.expander('Remote Sensing Processing:'):
    st.markdown('''Mean raw reflectance values of the **Red**, **Green**, **Blue**, **Near-Infrared(NIR)**, **Red-Edge** 
                wavelength bands were extracted from each plot at multiple dates in 2022 and 2023 with ArcPy pipelines.  
                [GIS_Tools - MSThreshAndExtract](https://github.com/B-Webster-Bio/GIS_Tools). To make the data more comparable 
                between years we will convert from sample date to DaysAfterPlanting(DAP). Next we will calculate 
                some well known vegetative indices with a focus on ones that self-normalize. 
                ''')
    st.subheader('Convert Dates to Days After Planting (DAP)')
    st.code(''' 
planting_date = '2023-05-25' # example
df = pd.DataFrame()
for f in d:
    df_n = pd.read_csv(f)
    df_n[['Temp', 'Band']] = df_n['Source'].str.split('_', expand=True)
    df_n['PlantingDate'] = pd.to_datetime('2023-05-25')
    sdate = f[:8]
    df_n['SampleDate'] = pd.to_datetime(sdate, format='%Y%m%d')
    df_n['DAP'] = (df_n['SampleDate'] - df_n['PlantingDate']).dt.days
    df = pd.concat([df, df_n])
    print(df_n.shape)
'''
            )
    st.subheader('Calculate Vegatative Indices')
    st.code('''
df_RS['NDVI'] = (df_RS['NIR'] - df_RS['Red']) / 
            (df_RS['NIR'] + df_RS['Red'])

df_RS['GNDVI'] = (df_RS['NIR'] - df_RS['Green']) / 
            (df_RS['NIR'] + df_RS['Green'])

df_RS['RDVI'] = (df_RS['NIR'] - df_RS['Red']) / 
            (np.sqrt(df_RS['NIR'] + df_RS['Red']))

df_RS['NLI'] = ((df_RS['NIR']**2) - df_RS['Red']) / 
            ((df_RS['NIR']**2) + df_RS['Red'])

df_RS['CVI'] = (df_RS['NIR'] * df_RS['NIR']) / 
            (df_RS['Green']**2)

df_RS['MSR'] = ((df_RS['NIR'] / df_RS['Red']) - 1) / 
            ((np.sqrt(df_RS['NIR'] / df_RS['Red'])) + 1)

df_RS['NDI'] = (df_RS['RedEdge'] - df_RS['Red']) / 
            (df_RS['RedEdge'] + df_RS['Red'])

df_RS['NDVIRedge'] = (df_RS['NIR'] - df_RS['RedEdge']) / 
            (df_RS['NIR'] + df_RS['RedEdge'])

df_RS['PSRI'] = (df_RS['Red'] - df_RS['Blue']) / 
            df_RS['RedEdge']

df_RS['CIRedge'] = (df_RS['NIR'] / 
            df_RS['RedEdge']) - 1

df_RS['MTCI'] = (df_RS['NIR'] - df_RS['RedEdge']) / 
            (df_RS['RedEdge'] - df_RS['Red'])
'''
            )
    df_ref = pd.merge(df_meta, unmelt, on =['PLOT', 'YEAR'], how = 'inner')
    df_ref = df_ref.loc[df_ref['PLOT_YEAR'].isin(plots2keep), ]
    df_ref['DAP'] = df_ref['DAP'].astype('int')

    def interpolate_spectral_indices(df):
        # Group by PLOT_YEAR and GENOTYPE
        grouped = df.groupby(['PLOT_YEAR', 'GENOTYPE'])
    
        # Initialize list to store interpolated dataframes
        interpolated_dfs = []
    
        # Common DAP range to interpolate to
        common_dap_range = range(40, 100)
    
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
                                     kind='cubic', fill_value='extrapolate')
            
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

    st.subheader('RS Coverage')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df_ref, x = 'DAP', y = 'NDVI', hue = 'YEAR', ax=ax, style='NTREATMENT', marker = 'x')
    plt.axvline(x=92, color='b', linestyle='--')
    plt.text(75, 0.5, 'Gas Ex Sampling 2022', 
         horizontalalignment='center', 
         verticalalignment='bottom')
    plt.axvline(x=99, color='orange', linestyle='--')
    plt.text(115, 0.4, 'Gas Ex Sampling 2023', 
         horizontalalignment='center', 
         verticalalignment='center')
    st.pyplot(fig)

    # interpolate raw data to same DAP
    df_ref_int = interpolate_spectral_indices(df_ref)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df_ref_int, x = 'DAP', y = 'NDVI', hue = 'YEAR', ax=ax, style='NTREATMENT', marker = 'x')
    st.pyplot(fig)


    st.markdown('''Remote sensing started earlier in 2022 and ended later in 2023 but otherwise there are no missing values. 
                These curves are the about expected shape from plants growing and then eventually dying off at the end 
                of the season. There is a clear difference between years so we should scale and standardize each date 
                within each year.''')

    spectral_columns = ['Blue', 'Green', 'NIR', 'Red', 'RedEdge', 'SAVIMASK', 
                    'NDVI', 'GNDVI', 'RDVI', 'NLI', 'CVI', 'MSR', 'NDI', 
                    'NDVIRedge', 'PSRI', 'CIRedge', 'MTCI']

    # Group by 'YEAR' and 'DAP', and normalize the spectral columns
    normalized_df = df_ref.copy()
    normalized_df[spectral_columns] = df_ref.groupby(['YEAR', 'DAP'])[spectral_columns].transform(
        lambda x: (x - x.mean()) / (x.std())
    )
    st.subheader('RS Coverage Norm 2022')
    df2022 = normalized_df.loc[normalized_df['YEAR'] == '2022']
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df2022, x = 'DAP', y = 'NDVI', hue = 'GENOTYPE', style='NTREATMENT', ax=ax, legend=False)
    st.pyplot(fig)

    st.subheader('RS Coverage Norm 2023')
    df2023 = normalized_df.loc[normalized_df['YEAR'] == '2023']
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df2023, x = 'DAP', y = 'NDVI', hue = 'GENOTYPE', style='NTREATMENT', ax=ax, legend=False)
    st.pyplot(fig)
    
    st.subheader('Interpolate common days between years')
    st.markdown('''Finally, let's interpolate values between 40 and 100 DAP with a cubic function. 
                 That will allow both years to be used for modelling.''')
    
    def interpolate_spectral_indices(df):
        # Group by PLOT_YEAR and GENOTYPE
        grouped = df.groupby(['PLOT_YEAR', 'GENOTYPE'])
    
        # Initialize list to store interpolated dataframes
        interpolated_dfs = []
    
        # Common DAP range to interpolate to
        common_dap_range = range(40, 100)
    
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
                                     kind='cubic', fill_value='extrapolate')
            
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
    interpolated_df = interpolate_spectral_indices(normalized_df)

    st.dataframe(interpolated_df)

    #normalized_df.to_csv('Data/NormMS.csv', index = False)

st.success("Saved Gas Ex and Agron data at Data/GasExAgron.csv")
st.success("Saved MultiSpec remote sensing data at Data/MS.csv")
st.success("Saved Normalized MultiSpec remote sensing data at Data/MS.csv")
