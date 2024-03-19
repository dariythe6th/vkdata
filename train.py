import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

train_data = pd.read_csv('train.csv')
features_data = pd.read_csv('features.csv')
train_data = pd.merge(train_data, features_data, on=['lat', 'lon'], how='left')

X = train_data.drop(['id', 'score'], axis=1)
y = train_data['score']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
val_predictions = model.predict(X_val)
mae = mean_absolute_error(y_val, val_predictions)
print("MAE на валидационном наборе данных:", mae)

joblib.dump(model, 'model.pkl')
