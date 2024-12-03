import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np

st.header("Explore Data")

def main():
    st.title('Agricultural Data Explorer')
    
    # Load data
    df = pd.read_csv(r'Data/GasExAgron.csv', index = False)
    
    # Sidebar for selecting graph type and variables
    st.sidebar.header('Graph Configuration')
    
    # Graph type selection
    graph_type = st.sidebar.selectbox(
        'Select Graph Type',
        ['Scatter Plot', 'Box Plot', 'Violin Plot']
    )
    
    # X-axis variable selection
    x_var = st.sidebar.selectbox(
        'X-Axis Variable',
        [col for col in df.columns if col not in ['PLOT_YEAR', 'YEAR']]
    )
    
    # Y-axis variable selection
    y_var = st.sidebar.selectbox(
        'Y-Axis Variable',
        [col for col in df.columns if col not in ['PLOT_YEAR', 'YEAR', x_var]]
    )
    
    # Color by selection
    color_var = st.sidebar.selectbox(
        'Color By',
        ['NTREATMENT', 'SUBPOPULATION', 'GENOTYPE']
    )
    
    # Create the plot based on selected graph type
    if graph_type == 'Scatter Plot':
        fig = px.scatter(
            df, 
            x=x_var, 
            y=y_var, 
            color=color_var, 
            title=f'{x_var} vs {y_var}'
        )
    elif graph_type == 'Box Plot':
        fig = px.box(
            df, 
            x=color_var, 
            y=y_var, 
            title=f'{y_var} by {color_var}'
        )
    else:  # Violin Plot
        fig = px.violin(
            df, 
            x=color_var, 
            y=y_var, 
            title=f'{y_var} Distribution by {color_var}'
        )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Basic statistics
    st.header('Data Summary')
    st.dataframe(df.describe())

if __name__ == '__main__':
    main()