import streamlit as st
import pandas as pd
import plotly.express as px

# Main Streamlit app
def main():
    st.title('Vegetative Index Time-Series Visualization')
    text = '''Take a look at some of the raw spectral index values over time by year and genotype.
    You might notice some systemmatice differences between years. We will try to control for that with
    filtering, imputation, and normalization in the _DataProcessing_ tab.'''
    st.markdown(text)
    # Load data
    df = pd.read_csv('Data/MS.csv')
    df['DAP'] = df['DAP'].astype('int')
    df = df.sort_values('DAP')
    # Sidebar for filtering
    st.sidebar.header('Filter Options')

    # VI Selection
    available_VI = sorted(['NIR', 'RedEdge',
    'NDVI', 'RDVI', 'NLI', 'CVI', 'MSR', 'NDI', 'NDVIRedge', 'PSRI', 'CIRedge', 'MTCI'
])
    selected_VI = st.selectbox('Select Vegetative Index:', available_VI)

    # Year selection
    available_years = sorted(df['YEAR'].unique())
    selected_years = st.sidebar.multiselect(
        'Select Years',
        available_years,
        default=available_years
    )

    # Genotype selection
    available_genotypes = sorted(df['GENOTYPE'].unique())
    selected_genotypes = st.sidebar.multiselect(
        'Select Genotypes',
        available_genotypes,
        default=['B73']
    )

    # Treatment selection
    available_treatments = sorted(df['NTREATMENT'].unique())
    selected_treatments = st.sidebar.multiselect(
        'Select Treatments',
        available_treatments,
        default=available_treatments
    )

    # Filter dataframe
    filtered_df = df[
        (df['YEAR'].isin(selected_years)) &
        (df['GENOTYPE'].isin(selected_genotypes)) &
        (df['NTREATMENT'].isin(selected_treatments))
    ]

    # Create line plot
    if not filtered_df.empty:
        fig = px.line(
            filtered_df,
            x='DAP',
            y=selected_VI,
            color='PLOT_YEAR',
            line_dash='NTREATMENT',
            title=f'{selected_VI} Over Days After Planting',
            labels={'DAP': 'Days After Planting', 'NDVI': 'Normalized Difference Vegetation Index'}
        )

        st.plotly_chart(fig, use_container_width=True)



    else:
        st.warning("No data matches the selected filters.")

if __name__ == '__main__':
    main()
