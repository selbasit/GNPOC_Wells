
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Load models and encoders
clf = joblib.load("well_type_classifier.pkl")
kmeans = joblib.load("well_kmeans.pkl")
encoder_block = joblib.load("encoder_block.pkl")
encoder_operator = joblib.load("encoder_operator.pkl")
encoder_target = joblib.load("encoder_target.pkl")
imputer_num = joblib.load("imputer_numeric.pkl")
imputer_cat = joblib.load("imputer_categorical.pkl")

st.set_page_config(page_title="GNPOC Well Analyzer", layout="wide")
st.title("🔍 GNPOC Well Intelligence App")

tab1, tab2 = st.tabs(["🎯 Predict WELL TYPE", "🧭 Cluster Explorer"])

with tab1:
    st.subheader("Predict the WELL TYPE")
    with st.form("prediction_form"):
        northing = st.number_input("NORTHING", format="%.2f")
        easting = st.number_input("EASTING", format="%.2f")
        msl = st.number_input("MSL (Mean Sea Level)", format="%.2f")
        block = st.text_input("BLOCK #")
        operator = st.text_input("OPERATOR")
        submit = st.form_submit_button("Predict")

    if submit:
        # Format input
        X_input = pd.DataFrame([[northing, easting, msl, block, operator]],
                               columns=["NORTHING", "EASTING", "MSL", "BLOCK #", "OPERATOR"])
        # Impute missing if any
        X_input[["NORTHING", "EASTING", "MSL"]] = imputer_num.transform(X_input[["NORTHING", "EASTING", "MSL"]])
        X_input[["BLOCK #", "OPERATOR"]] = imputer_cat.transform(X_input[["BLOCK #", "OPERATOR"]])

        # Encode
        X_input["BLOCK #"] = encoder_block.transform(X_input["BLOCK #"])
        X_input["OPERATOR"] = encoder_operator.transform(X_input["OPERATOR"])

        # Predict
        pred_encoded = clf.predict(X_input)[0]
        pred_label = encoder_target.inverse_transform([pred_encoded])[0]

        st.success(f"✅ Predicted WELL TYPE: **{pred_label}**")

with tab2:
    st.subheader("Clustering of Wells (KMeans)")
    st.markdown("This visualization shows clusters based on spatial and elevation features.")

    # Sample fake data for visualization
    import os
    if os.path.exists("sample_data.csv"):
        df = pd.read_csv("sample_data.csv")
    else:
        # Generate placeholder
        df = pd.DataFrame({
            "EASTING": np.random.uniform(750000, 800000, 100),
            "NORTHING": np.random.uniform(1000000, 1150000, 100),
            "Cluster": kmeans.predict(np.column_stack([
                np.random.uniform(1000000, 1150000, 100),
                np.random.uniform(750000, 800000, 100),
                np.random.uniform(390, 420, 100),
                np.random.randint(0, 5, 100),
                np.random.randint(0, 3, 100)
            ]))
        })

    fig = px.scatter(df, x="EASTING", y="NORTHING", color=df["Cluster"].astype(str),
                     title="KMeans Clustering of Wells",
                     labels={"color": "Cluster ID"}, height=600)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


import pyproj
from pyproj import Transformer

with st.tabs(["🎯 Predict WELL TYPE", "🧭 Cluster Explorer", "📍 Predict GPS Location"])[2]:
    st.subheader("Predict Real-World Coordinates (LAT/LON)")
    st.markdown("Use EASTING/NORTHING and other metadata to estimate well's GPS position.")

    with st.form("gps_form"):
        northing = st.number_input("NORTHING", format="%.2f", key="gps_northing")
        easting = st.number_input("EASTING", format="%.2f", key="gps_easting")
        msl = st.number_input("MSL (Mean Sea Level)", format="%.2f", key="gps_msl")
        block = st.text_input("BLOCK #", key="gps_block")
        operator = st.text_input("OPERATOR", key="gps_operator")
        submit_gps = st.form_submit_button("Estimate Location")

    if submit_gps:
        # Prepare input
        X_input = pd.DataFrame([[northing, easting, msl, block, operator]],
                               columns=["NORTHING", "EASTING", "MSL", "BLOCK #", "OPERATOR"])

        # Handle missing/categorical encoding
        X_input[["NORTHING", "EASTING", "MSL"]] = imputer_num.transform(X_input[["NORTHING", "EASTING", "MSL"]])
        X_input[["BLOCK #", "OPERATOR"]] = imputer_cat.transform(X_input[["BLOCK #", "OPERATOR"]])

        # Load regression encoders
        block_encoder_reg = joblib.load("block_encoder_reg.pkl")
        op_encoder_reg = joblib.load("op_encoder_reg.pkl")

        X_input["BLOCK #"] = block_encoder_reg.transform(X_input["BLOCK #"])
        X_input["OPERATOR"] = op_encoder_reg.transform(X_input["OPERATOR"])

        # Predict coordinates
        lat_model = joblib.load("lat_model.pkl")
        lon_model = joblib.load("lon_model.pkl")
        pred_lat = lat_model.predict(X_input)[0]
        pred_lon = lon_model.predict(X_input)[0]

        # Convert EASTING/NORTHING using UTM
        transformer = Transformer.from_crs("epsg:32636", "epsg:4326", always_xy=True)
        lon_utm, lat_utm = transformer.transform(easting, northing)

        st.success(f"📍 Predicted Coordinates (ML): **{pred_lat:.6f}°, {pred_lon:.6f}°**")
        st.info(f"📐 UTM Converted Coordinates: **{lat_utm:.6f}°, {lon_utm:.6f}°**")
    