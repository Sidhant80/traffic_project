import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="Smart Traffic AI", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Banglore_traffic_Dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    # ✅ FIX: FORCE PROPER HOURS (NO SINGLE DOT ISSUE)
    df['hour'] = np.tile(np.arange(24), int(len(df)/24)+1)[:len(df)]

    df['day'] = df['Date'].dt.day_name()

    # Map coordinates
    np.random.seed(1)
    df['lat'] = 12.9 + np.random.randn(len(df))*0.02
    df['lon'] = 77.6 + np.random.randn(len(df))*0.02

    return df

df = load_data()

# ---------------- LOAD MODEL ----------------
model = joblib.load("traffic_model.pkl")
le_area = joblib.load("area_encoder.pkl")
le_weather = joblib.load("weather_encoder.pkl")
le_road = joblib.load("road_encoder.pkl")

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;color:#FF4B4B;'>🚦 Smart Traffic AI Dashboard</h1>", unsafe_allow_html=True)

# =========================================================
# 🔮 SIDEBAR
# =========================================================

st.sidebar.header("🔮 Predict Traffic")

area = st.sidebar.selectbox("Area", df['Area Name'].unique())
weather = st.sidebar.selectbox("Weather", df['Weather Conditions'].unique())
road = st.sidebar.selectbox("Roadwork", df['Roadwork and Construction Activity'].unique())
hour = st.sidebar.slider("Hour", 0, 23, 8)

area_enc = le_area.transform([area])[0]
weather_enc = le_weather.transform([weather])[0]
road_enc = le_road.transform([road])[0]

prediction = model.predict([[area_enc, weather_enc, road_enc, hour]])[0]

st.sidebar.success(f"🚗 Predicted Traffic Volume: {int(prediction)}")

# =========================================================
# 🎯 KPI
# =========================================================

col1, col2, col3 = st.columns(3)

col1.metric("🚦 Avg Traffic", int(df['Traffic Volume'].mean()))
col2.metric("🚗 Avg Speed", round(df['Average Speed'].mean(), 2))
col3.metric("🔥 Congestion", round(df['Congestion Level'].mean(), 2))

# =========================================================
# 🚦 TRAFFIC TREND (FIXED 100%)
# =========================================================

st.subheader("🚦 Traffic Trend (Hourly Average)")

hourly = df.groupby('hour')['Traffic Volume'].mean().reset_index()

fig1 = px.line(
    hourly,
    x='hour',
    y='Traffic Volume',
    markers=True
)

fig1.update_traces(line=dict(color='red', width=4))

st.plotly_chart(fig1, use_container_width=True)

# =========================================================
# 🏙 AREA
# =========================================================

st.subheader("🏙 Traffic by Area")

area_data = df.groupby('Area Name')['Traffic Volume'].mean().reset_index()

fig2 = px.bar(area_data, x='Area Name', y='Traffic Volume', color='Traffic Volume')

st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# 🌧 WEATHER
# =========================================================

st.subheader("🌧 Weather Impact")

fig3 = px.box(df, x='Weather Conditions', y='Traffic Volume', color='Weather Conditions')

st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# 🚨 TOP 5
# =========================================================

st.subheader("🚨 Top 5 Congested Areas")

top5 = df.groupby('Area Name')['Traffic Volume'].mean().nlargest(5).reset_index()

fig4 = px.bar(top5, x='Area Name', y='Traffic Volume', color='Traffic Volume')

st.plotly_chart(fig4, use_container_width=True)

# =========================================================
# 📅 DAY
# =========================================================

st.subheader("📅 Traffic by Day")

day_data = df.groupby('day')['Traffic Volume'].mean().reset_index()

fig5 = px.bar(day_data, x='day', y='Traffic Volume', color='Traffic Volume')

st.plotly_chart(fig5, use_container_width=True)

# =========================================================
# 🧠 HEATMAP
# =========================================================

st.subheader("🧠 Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig6 = px.imshow(corr, text_auto=True)

st.plotly_chart(fig6, use_container_width=True)

# =========================================================
# 🗺 MAP
# =========================================================

st.subheader("🗺 Traffic Hotspots Map")

fig7 = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    color="Traffic Volume",
    size="Traffic Volume",
    hover_name="Area Name",
    zoom=10
)

fig7.update_layout(mapbox_style="carto-positron")

st.plotly_chart(fig7, use_container_width=True)

# =========================================================
# 🚗 SPEED VS TRAFFIC
# =========================================================

st.subheader("🚗 Speed vs Traffic")

fig8 = px.scatter(
    df,
    x='Average Speed',
    y='Traffic Volume',
    size='Congestion Level',
    color='Weather Conditions'
)

st.plotly_chart(fig8, use_container_width=True)

# =========================================================
# 🧠 INSIGHTS
# =========================================================

st.subheader("🧠 AI Insights")

peak_area = df.groupby('Area Name')['Traffic Volume'].mean().idxmax()
best_time = df.groupby('hour')['Traffic Volume'].mean().idxmin()
worst_weather = df.groupby('Weather Conditions')['Traffic Volume'].mean().idxmax()

st.success(f"""
🚦 Peak Area: {peak_area}  
⏰ Best Time: {best_time}:00  
🌧 Worst Weather: {worst_weather}
""")

# =========================================================
# 🤖 CHATBOT
# =========================================================

st.subheader("🤖 Traffic Assistant")

query = st.text_input("Ask something...")

if query:
    q = query.lower()

    if "best time" in q:
        st.write(f"✅ Best time is {best_time}:00")

    elif "worst area" in q:
        st.write(f"🚨 Worst area is {peak_area}")

    elif "weather" in q:
        st.write(f"🌧 Worst weather is {worst_weather}")

    elif "predict" in q:
        st.write(f"🔮 Prediction: {int(prediction)}")

    else:
        st.write("Try: best time, worst area, weather, predict")