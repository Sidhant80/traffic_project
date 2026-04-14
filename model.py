import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("Banglore_traffic_Dataset.csv")
df.dropna(inplace=True)

# Date processing
df['Date'] = pd.to_datetime(df['Date'])
df['hour'] = df['Date'].dt.hour.fillna(0)

# Encoding
le_area = LabelEncoder()
le_weather = LabelEncoder()
le_road = LabelEncoder()

df['Area Name'] = le_area.fit_transform(df['Area Name'])
df['Weather Conditions'] = le_weather.fit_transform(df['Weather Conditions'])
df['Roadwork and Construction Activity'] = le_road.fit_transform(
    df['Roadwork and Construction Activity']
)

# Features & target
X = df[['Area Name', 'Weather Conditions', 'Roadwork and Construction Activity', 'hour']]
y = df['Traffic Volume']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model + encoders
joblib.dump(model, "traffic_model.pkl")
joblib.dump(le_area, "area_encoder.pkl")
joblib.dump(le_weather, "weather_encoder.pkl")
joblib.dump(le_road, "road_encoder.pkl")

print("✅ Model trained successfully!")