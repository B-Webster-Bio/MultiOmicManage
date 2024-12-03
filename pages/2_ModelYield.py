import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np

st.header("Model Yield")
st.markdown('Leverage multi-omics to predict plot yield without harvesting.')