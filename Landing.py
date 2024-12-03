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
    st.image(r'Supp/CornHarvest.PNG', caption = 'Hand corn harvest (PBS Wisconsin)')

with col2:
    st.markdown('Agronomic traits were collected by hand. They include some of the most important trait that a plant breeding might consider:')
    st.markdown("**KernelDryWt_PerPlant** -  Avg grams of grain per plant harvested")
    st.markdown('**DaysToSilk** - The number of days after planting it takes for flowering to occur')
    st.markdown('**AvgFlagHt** - The height measured from the ground to the top leaf')

st.subheader('2. Leaf gas exchange')
col1, col2 = st.columns(2)
with col1:
    st.image('Supp/LeafGasEx.png', caption = 'Gas exchange through leaf stomata')

with col2:
    st.header("Gas exchange parameters")
    st.markdown("* A - CO2 assimilation rate (µmol CO2 m⁻² s⁻¹)")
    st.markdown("* E - Transpiration of H2O (mol H2O m⁻² s⁻¹)")
    st.markdown("* gsw - stomatal conductance to H2O (mol H2O m⁻² s⁻¹)")
    st.markdown("* Ci - interceullar CO2 concentration ready for assimilation (ppm)")

st.subheader('Gas exchange can be measured by a Licor')
st.image('Supp/Licor.PNG', caption = 'Licor 6800 measuring plant leaf in the field')