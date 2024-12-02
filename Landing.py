import streamlit as st

st.title('Multi-omic Management for Plant Breeding')
md = '''The plant breeding process makes extensive use of data science techniques to identify and select useful plant germplasm. 
In recent years there has been an explosion of the data types available to plant breeders. From remote sensing to advances in plant health monitoring. The abundance of data can be helpful but also 
hard to manage. Multi-omic Mangement aims to provide a resource that can help integrate and leverage multi-omic data in a plant breeding context.'''
st.markdown(md)

st.title('Data Types:')
st.subheader('1. Agronomic')
col1, col2 = st.columns(2)
with col1:
    st.image('Supp/CornHarvest.png', caption = 'Hand corn harvest (PBS Wisconsin)')

with col2:
    st.markdown("KernelDryWt_PerPlant -  Avg grams of grain per plant")