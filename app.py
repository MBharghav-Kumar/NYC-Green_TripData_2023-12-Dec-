import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title=" NYC Green Taxi Fare Predictor", layout="wide")


st.title(" NYC Green Taxi Fare Predictor")
st.markdown("Use **Multiple Linear Regression** to estimate taxi fare (`total_amount`) based on trip data from NYC Green Taxi 2023-12.\n\n---")


@st.cache_data
def load_data():
    df = pd.read_parquet(r"c:\Users\ASUS\Downloads\bharghav\green_tripdata_2023-12.parquet")
    df["trip_duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna("Unknown", inplace=True)
    df["weekday"] = df["lpep_dropoff_datetime"].dt.day_name()
    df["hourofday"] = df["lpep_dropoff_datetime"].dt.hour
    df = df[df['total_amount'] >= 0]

    # Encode categorical variables
    df_encoded = pd.get_dummies(df[["store_and_fwd_flag", "RatecodeID", "payment_type", "trip_type", "weekday", "hourofday"]], drop_first=True)

    # Feature selection
    numeric_cols = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 
                    'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 
                    'trip_duration', 'passenger_count']

    X = pd.concat([df[numeric_cols], df_encoded], axis=1)
    y = df["total_amount"]
    
    return df, X, y

df, X, y = load_data()


top_10_features = X.corrwith(y).abs().sort_values(ascending=False).head(10).index.tolist()


@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X[top_10_features], y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model(X, y)

st.sidebar.title(" Feature Input")
st.sidebar.markdown("Enter values for the top 10 features used in prediction:")

default_values = {
    'trip_distance': 2.0,
    'fare_amount': 8.0,
    'extra': 0.5,
    'mta_tax': 0.5,
    'tip_amount': 1.5,
    'tolls_amount': 0.0,
    'improvement_surcharge': 0.3,
    'congestion_surcharge': 2.5,
    'trip_duration': 10.0,
    'passenger_count': 1.0,
}

user_input = {}
for col in top_10_features:
    val = st.sidebar.number_input(f"{col}", value=default_values.get(col, 0.0))
    user_input[col] = val

input_df = pd.DataFrame([user_input])

col1, col2 = st.columns(2)

with col1:
    st.subheader(" Predict Fare")
    if st.button(" Predict Total Amount"):
        input_df = input_df[top_10_features]
        prediction = model.predict(input_df)[0]
        prediction = max(0, prediction)
        st.success(f" Predicted Total Amount: **${prediction:.2f}**")

with col2:
    st.subheader(" Top 10 Features Used")
    st.write(pd.DataFrame({"Feature": top_10_features}).style.hide_index())


st.markdown("---")
st.subheader(" Visualize `total_amount` Distribution")

plot_type = st.selectbox("Choose Visualization Type", ["Histogram", "Boxplot", "Density Curve"])
fig, ax = plt.subplots(figsize=(10, 4))

if plot_type == "Histogram":
    sns.histplot(df['total_amount'], bins=50, kde=False, ax=ax)
    ax.set_title("Histogram of Total Amount")
    ax.set_xlabel("Total Amount")
    ax.set_ylabel("Frequency")
elif plot_type == "Boxplot":
    sns.boxplot(x=df['total_amount'], ax=ax)
    ax.set_title("Boxplot of Total Amount")
    ax.set_xlabel("Total Amount")
elif plot_type == "Density Curve":
    sns.kdeplot(df['total_amount'], fill=True, ax=ax)
    ax.set_title("Density Curve of Total Amount")
    ax.set_xlabel("Total Amount")

st.pyplot(fig)
