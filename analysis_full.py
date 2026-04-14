# =========================================================
# TRAFFIC ANALYSIS PROJECT - FINAL IMPROVED CODE
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Banglore_traffic_Dataset.csv")

# ---------------- CLEAN DATA ----------------
df.dropna(inplace=True)

# ---------------- DATE PROCESSING ----------------
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['day'] = df['Date'].dt.day_name()

# ---------------- CREATE REALISTIC HOUR ----------------
df['hour'] = df['Date'].dt.hour.fillna(0)

# ---------------- CREATE TRAFFIC CATEGORY ----------------
def traffic_level(x):
    if x < 200:
        return "Low"
    elif x < 500:
        return "Medium"
    else:
        return "High"

df['Traffic Category'] = df['Traffic Volume'].apply(traffic_level)

# =========================================================
# VISUALIZATION
# =========================================================

# Traffic by Area
plt.figure()
df['Area Name'].value_counts().plot(kind='bar')
plt.title("Traffic Density by Area")
plt.xticks(rotation=45)
plt.show()

# Traffic by Day
plt.figure()
sns.countplot(x='day', data=df)
plt.xticks(rotation=45)
plt.title("Traffic by Day")
plt.show()

# Traffic by Hour
plt.figure()
sns.lineplot(x='hour', y='Traffic Volume', data=df)
plt.title("Traffic Trend by Hour")
plt.show()

# Weather Impact
plt.figure()
sns.boxplot(x='Weather Conditions', y='Traffic Volume', data=df)
plt.xticks(rotation=45)
plt.title("Weather Impact on Traffic")
plt.show()

# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =========================================================
# MACHINE LEARNING (CORRECTED)
# =========================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

ml_df = df.copy()

le = LabelEncoder()

ml_df['Weather Conditions'] = le.fit_transform(ml_df['Weather Conditions'])
ml_df['Roadwork and Construction Activity'] = le.fit_transform(
    ml_df['Roadwork and Construction Activity']
)

ml_df['Traffic Category'] = le.fit_transform(ml_df['Traffic Category'])

# FEATURES
X = ml_df[['Average Speed', 'Congestion Level', 'Weather Conditions']]
y = ml_df['Traffic Category']

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MODEL
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ACCURACY
accuracy = model.score(X_test, y_test)

print("MODEL ACCURACY:", accuracy)

print("\nPROJECT COMPLETED SUCCESSFULLY")