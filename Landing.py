import streamlit as st

st.title('Multi-omic Management for Plant Breeding')
md = '''The plant breeding process makes extensive use of data science techniques to identify and select useful plant germplasm. 
In recent years there has been an explosion of the data types available to plant breeders. From remote sensing to advances in plant health monitoring. The abundance of data can be helpful but also 
hard to manage. Multi-omic Mangement aims to provide a resource that can help integrate and leverage multi-omic data in a plant breeding context.'''
st.markdown(md)

st.title('Data Types and Description:')
st.subheader('1. Agronomic')
col1, col2 = st.columns(2)
with col1:
    st.image('Supp/CornHarvest.PNG', caption = 'Hand corn harvest (PBS Wisconsin)')

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
    st.markdown('Gas ex traits measure how much CO2 and oxygen plants are "Breathing" which is in turn related to their metabolism.')
    st.markdown("**A** - CO2 assimilation rate (µmol CO2 m⁻² s⁻¹)")
    st.markdown("**E** - Transpiration of H2O (mol H2O m⁻² s⁻¹)")
    st.markdown("**gsw** - stomatal conductance to H2O (mol H2O m⁻² s⁻¹)")
    st.markdown("**Ci** - interceullar CO2 concentration ready for assimilation (ppm)")

st.image('Supp/Licor.PNG', caption = 'Licor 6800 measuring gas ex. in the field')

st.subheader('3. Remote Sensing')
col1, col2 = st.columns(2)
with col1:
    st.image('Supp/Drone.jpg', caption = 'DJI drone used to collect multispectral imagery')
    st.image('Supp/NDVI.png', caption = 'NDVI is generally related to chlorophyll content')

with col2:
    st.markdown('''Reflectance values of red, green, blue, near-infrared(NIR), and red-edge wavelengths are together called multispectral imagery. 
                They can be collected quickly by flying a drone over a field and then processed into vegetative indices such as NDVI. Veg indices can be
                highly associated with other traits of interest such as yield. One advantage of remote sensing data is that it provides a nice time-series 
                data set that can span much of the growing season.''')
    st.markdown("**Vegetative Indices**:")
    st.markdown('''**NDVI**, **GNDVI**, **RDVI**,  
                 **NLI**, **CVI**, **MSR**,  
                **NDI**, **NDVIRedEdge**, **PSRI**,  
                 **CIRedEdge**, **MTCI**''')
    

st.image('Supp/RGBFIELD.png', caption = 'RGB image from drone')
st.image('Supp/SAVIFIELD.JPG', caption = 'NDVI image from drone')