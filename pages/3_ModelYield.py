import pandas as pd
import numpy as np
from scipy import interpolate
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df_NormRef = pd.read_csv('Data/NormMS_Interp.csv')
df_agron = pd.read_csv('Data/GasExAgron.csv')

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

df_all = pd.merge(df_agron, wide_df, on=['PLOT_YEAR', 'YEAR', 'GENOTYPE', 'PLOT', 'NTREATMENT'])
df_all = df_all.dropna(axis=1)

st.markdown('If we flatten out the 16 Remote sensing spectra across 60 time points we have generated 16 * 60 = 960 features that can help predict yield')
st.dataframe(df_all)

s1 = list(df_all.columns[6:10])
s2 = list(df_all.columns[11:])
all_possible_features = s1 + s2
agron_features = all_possible_features[:6]
st.markdown('**All possible features**')
st.write(all_possible_features)


st.write(df_all[all_possible_features])


y = df_all['KERNELDRYWT_PERPLANT']
X = df_all[all_possible_features]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix objects for XGBoost (optional but recommended for optimization)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for XGBoost
params = {
    'objective': 'reg:squarederror',  # Regression task
    'eval_metric': 'rmse',
    'eta': 0.1,  # Learning rate
    'max_depth': 6,  # Depth of the trees
    'seed': 42
}

# Train the XGBoost model
evals = [(dtrain, 'train'), (dtest, 'test')]
xgb_model = xgb.train(params, dtrain, num_boost_round=50, evals=evals, early_stopping_rounds=10)

# Predict and calculate RMSE on the test set
y_pred = xgb_model.predict(dtest)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Streamlit App
st.title("XGBoost Feature Importance Visualization")

# Display Test RMSE
st.write(f"Test RMSE: {rmse:.4f}")

# Function to plot and display feature importance
def plot_and_display_importance(importance_type, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(xgb_model, importance_type=importance_type, ax=ax, title=title)
    st.pyplot(fig)

# Dropdown to select importance type
importance_type = st.selectbox(
    "Select Feature Importance Type",
    ["weight", "gain", "cover"],
    format_func=lambda x: x.capitalize()
)

# Plot and display the selected importance type
plot_and_display_importance(importance_type, f"Feature Importance ({importance_type.capitalize()})")