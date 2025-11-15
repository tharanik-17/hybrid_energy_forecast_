
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os

st.set_page_config(layout='wide')
st.title('Hybrid Renewable Energy Dashboard - Demo')

DATA_PATH = 'data/processed/cleaned_hybrid_energy_sample.csv'
if os.path.exists('data/processed/cleaned_hybrid_energy.csv'):
    DATA_PATH = 'data/processed/cleaned_hybrid_energy.csv'  # prefer user's cleaned file if present

df = pd.read_csv(DATA_PATH, parse_dates=['DATE_TIME'])
st.sidebar.write('Data range: {} to {}'.format(df['DATE_TIME'].min().date(), df['DATE_TIME'].max().date()))

# Quick preview
st.subheader('Data Preview')
st.dataframe(df.head(100))

# Time series plot
st.subheader('AC Power over time')
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df['DATE_TIME'], df['AC_POWER'], label='AC_POWER')
ax.set_xlabel('Date'); ax.set_ylabel('AC Power')
st.pyplot(fig)

# Scatter AC_POWER vs IRRADIATION
st.subheader('AC Power vs Irradiation')
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.scatter(df['IRRADIATION'], df['AC_POWER'], s=6)
ax2.set_xlabel('Irradiation'); ax2.set_ylabel('AC Power')
st.pyplot(fig2)

# If model exists, show predictions
st.subheader('Model Predictions (if available)')
model_path = 'models/xgboost_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    df['hour'] = df['DATE_TIME'].dt.hour
    df['weekday'] = df['DATE_TIME'].dt.weekday
    df['AC_roll_3'] = df['AC_POWER'].rolling(3, min_periods=1).mean()
    features = ['DC_POWER','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION','hour','weekday','AC_roll_3']
    df2 = df.dropna(subset=features)
    preds = model.predict(df2[features])
    df2['pred_xgb'] = preds
    st.write(df2[['DATE_TIME','AC_POWER','pred_xgb']].head(50))
    fig3, ax3 = plt.subplots(figsize=(12,4))
    ax3.plot(df2['DATE_TIME'].iloc[:200], df2['AC_POWER'].iloc[:200], label='Actual')
    ax3.plot(df2['DATE_TIME'].iloc[:200], df2['pred_xgb'].iloc[:200], label='XGB Pred')
    ax3.legend()
    st.pyplot(fig3)
else:
    st.info('No model found at models/xgboost_model.pkl. Run src/train_and_save.py to create one.')

st.sidebar.info('Replace the sample CSV with your cleaned_hybrid_energy.csv in data/processed/ for full dataset.')
