import pandas as pd
import numpy as np

def load_and_clean_data():
    df = pd.read_csv("Banglore_traffic_Dataset.csv")

    df = df.dropna()

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['day'] = df['Date'].dt.day_name()

    np.random.seed(0)
    df['hour'] = np.random.randint(0, 24, size=len(df))

    return df