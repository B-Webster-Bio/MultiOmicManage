import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np
from scipy import stats
import plotly.graph_objects as go

st.header("Data Explorer")
st.markdown('')

def main():
    st.title('Agricultural Data Explorer')
    
    # Load data
    df = pd.read_csv('Data/GasExAgron.csv')
    df['YEAR'] = df['YEAR'].astype('str')
    df['PLOT'] = df['PLOT'].astype('str')
    exclude_cols = ['NTREATMENT', 'SUBPOPULATION', 'GENOTYPE', 'YEAR', 'PLOT_YEAR', 'PLOT']
    quant_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Sidebar for selecting graph type and variables
    st.sidebar.header('Graph Configuration')
    
    # Graph type selection
    graph_type = st.sidebar.selectbox(
        'Select Graph Type',
        ['Scatter Plot with Regression']
    )
    
    # X-axis variable selection
    x_var = st.sidebar.selectbox(
        'X-Axis Variable',
        quant_cols
    )
    
    # Y-axis variable selection
    y_var = st.sidebar.selectbox(
        'Y-Axis Variable',
        [col for col in quant_cols if col != x_var]
    )
    
    # Color by selection
    color_var = st.sidebar.selectbox(
        'Color By',
        ['NTREATMENT', 'SUBPOPULATION', 'GENOTYPE', 'YEAR']
    )
    
    # Create scatter plot with regression line
    fig = go.Figure()
    
    # Add scatter points
    for category in df[color_var].unique():
        subset = df[df[color_var] == category]
        fig.add_trace(go.Scatter(
            x=subset[x_var], 
            y=subset[y_var], 
            mode='markers',
            name=category,
            text=subset['PLOT']  # Optional: add plot number as hover text
        ))
    
    # Calculate regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_var], df[y_var])
    
    # Add regression line
    x_range = np.linspace(df[x_var].min(), df[x_var].max(), 100)
    y_range = intercept + slope * x_range
    fig.add_trace(go.Scatter(
        x=x_range, 
        y=y_range, 
        mode='lines', 
        name='Regression Line',
        line=dict(color='red', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{y_var} vs {x_var}',
        xaxis_title=x_var,
        yaxis_title=y_var
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display R-squared value
    st.write(f"R-squared: {r_value**2:.4f}")
    st.write(f"p-value {p_value}")
    st.write(f"Regression Equation: {y_var} = {slope:.4f} * {x_var} + {intercept:.4f}")

    # histograms
   # Create histogram
    fig = go.Figure()
    for group in df[color_var].unique():
        group_data = df[df[color_var] == group][x_var]
        fig.add_trace(go.Histogram(
            x=group_data, 
            name=group,  # This creates the legend entry
            opacity=0.7
        ))
    # Customize layout
    fig.update_layout(
        title=x_var,
        xaxis_title='Value',
        yaxis_title='Frequency'
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    for group in df[color_var].unique():
        group_data = df[df[color_var] == group][y_var]
        fig.add_trace(go.Histogram(
            x=group_data, 
            name=group,  # This creates the legend entry
            opacity=0.7
        ))
    # Customize layout
    fig.update_layout(
        title=y_var,
        xaxis_title='Value',
        yaxis_title='Frequency'
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df)
if __name__ == '__main__':
    main()