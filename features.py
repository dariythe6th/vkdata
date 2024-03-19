import pandas as pd
from geopy.distance import geodesic

train_data = pd.read_csv('train.csv')
features_data = pd.read_csv('features.csv')

train_data = pd.merge(train_data, features_data, on=['lat', 'lon'], how='left')

# Генерация новых признаков
# расстояние до определенной точки (координаты)
def calculate_distance(row, point_coords):
    object_coords = (row['lat'], row['lon'])
    distance = geodesic(object_coords, point_coords).kilometers
    return distance

# расстояние до центра города
city_center_coords = (latitude_of_city_center, longitude_of_city_center)
train_data['distance_to_city_center'] = train_data.apply(lambda row: calculate_distance(row, city_center_coords), axis=1)

# количество объектов ритейла в радиусе
def count_retail_objects_in_radius(row, retail_data, radius):
    object_coords = (row['lat'], row['lon'])
    nearby_objects = retail_data[retail_data.apply(lambda x: geodesic(object_coords, (x['lat'], x['lon'])).kilometers <= radius, axis=1)]
    return len(nearby_objects)

# количество объектов ритейла в радиусе 1 км
radius = 1
train_data['num_retail_objects_in_radius'] = train_data.apply(lambda row: count_retail_objects_in_radius(row, features_data, radius), axis=1)

# Сохранение данных с новыми признаками
train_data.to_csv('train_with_features.csv', index=False)
