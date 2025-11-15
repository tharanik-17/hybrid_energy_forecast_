
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

df = pd.read_csv('data/processed/cleaned_hybrid_energy_sample.csv', parse_dates=['DATE_TIME'])
df['hour'] = df['DATE_TIME'].dt.hour
df['weekday'] = df['DATE_TIME'].dt.weekday
df['AC_roll_3'] = df['AC_POWER'].rolling(3, min_periods=1).mean()

features = ['DC_POWER','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION','hour','weekday','AC_roll_3']
df = df.dropna(subset=features + ['AC_POWER'])

X = df[features]
y = df['AC_POWER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6)
model.fit(X_train, y_train)

joblib.dump(model, 'models/xgboost_model.pkl')
print('Saved models/xgboost_model.pkl')
