import pandas as pd
import numpy as np
from scipy import interpolate
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
RS_Features = all_possible_features[7:]
st.markdown('**All possible features**')
st.write(all_possible_features)


st.write(df_all[all_possible_features])


# Streamlit App
st.title("XGBoost and Linear Regression Feature Importance and Comparison")

# Feature set selection
feature_set_options = {
    "All Possible Features": all_possible_features,
    "Agronomic": agron_features,
    "RemoteSeing (Takes Long Time)": RS_Features
}

selected_feature_set_name = st.selectbox(
    "Select Feature Set",
    options=list(feature_set_options.keys()),
)

selected_features = feature_set_options[selected_feature_set_name]

# Extract features (X) and target (y)
y = df_all['KERNELDRYWT_PERPLANT']
X = df_all[selected_features]

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
xgb_model = xgb.train(params, dtrain, num_boost_round=10, evals=evals, early_stopping_rounds=10)

# Predict and calculate RMSE for XGBoost
y_pred_xgb = xgb_model.predict(dtest)
rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)

# Extract feature importance
importance = xgb_model.get_score(importance_type='weight')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
important_features = [feat[0] for feat in sorted_importance]

# RMSE from XGBoost
st.write(f"Test RMSE (XGBoost): {rmse_xgb:.4f}")

# Select top 'n' features
top_n = st.slider("Select Top n Features", min_value=1, max_value=len(important_features), value=10)
selected_top_features = important_features[:top_n]

# Filter data based on top 'n' features
X_train_filtered = X_train[selected_top_features]
X_test_filtered = X_test[selected_top_features]

# Train Linear Regression model on filtered data
linear_model = LinearRegression()
linear_model.fit(X_train_filtered, y_train)
y_pred_lr = linear_model.predict(X_test_filtered)
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)

# RMSE from Linear Regression
st.write(f"Test RMSE (Linear Regression): {rmse_lr:.4f}")

# Function to plot and display feature importance
def plot_and_display_importance(title, top_n):
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_features = sorted_importance[:top_n]
    feature_names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    ax.barh(feature_names[::-1], values[::-1], color='skyblue')
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

# Plot and display the selected importance type for top `n` features
plot_and_display_importance(f"Top {top_n} Feature Importance (Weight)", top_n)

# Comparison Section
st.write("### Comparison Summary")
st.write(
    f"Using the **{selected_feature_set_name}** feature set with the top {top_n} features:\n"
    f"- **XGBoost RMSE**: {rmse_xgb:.4f}\n"
    f"- **Linear Regression RMSE**: {rmse_lr:.4f}\n"
    f"XGBoost is typically better for non-linear relationships, while Linear Regression excels with linear patterns."
)