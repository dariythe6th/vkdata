import pandas as pd
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Загрузка данных
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
features_data = pd.read_csv('features.csv')

# Объединение признаков
train_data = pd.merge(train_data, features_data, on=['lat', 'lon'], how='left')
test_data = pd.merge(test_data, features_data, on=['lat', 'lon'], how='left')

# Разделение данных на признаки и целевую переменную
X_train = train_data.drop(['id', 'score'], axis=1)
y_train = train_data['score']
X_test = test_data.drop('id', axis=1)

# Выбор и обучение модели
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание на тестовом наборе данных
test_predictions = model.predict(X_test)

# Создание файла submission.csv
submission_df = pd.DataFrame({'id': test_data['id'], 'score': test_predictions})
submission_df.to_csv('submission.csv', index=False)
