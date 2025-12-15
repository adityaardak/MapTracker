import streamlit as st
import pandas as pd
import numpy as np
import json
from io import StringIO
import streamlit.components.v1 as components

from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="GPS Analytics & Unsupervised ML", layout="wide")

# ------------------------------------------------------------------
# SAMPLE DATA
# ------------------------------------------------------------------
SAMPLE_TSV = """time\tseconds_elapsed\tbearingAccuracy\tspeedAccuracy\tverticalAccuracy\thorizontalAccuracy\tspeed\tbearing\taltitude\tlongitude\tlatitude
1.74296E+18\t0.255999756\t45\t1.5\t1.307455301\t20.59600067\t0\t0\t374.8000183\t78.5795227\t21.2710244
1.74296E+18\t3.175305176\t56.90444946\t0.587098598\t1.336648345\t12.78999996\t0.530843079\t102.8688431\t374.8000183\t78.5796365\t21.2710753
1.74296E+18\t3.242290771\t71.24659729\t0.923150122\t1.337318182\t10.19900036\t0.546774924\t102.1160355\t374.8000183\t78.5796392\t21.2710753
1.74296E+18\t4.241882813\t89.88391113\t0.781779766\t1.347314119\t8.765999794\t0.213110328\t88.08830261\t374.8000183\t78.5796205\t21.2710833
"""

# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
def load_data(uploaded, sheet_url):
    if uploaded is not None:
        if uploaded.name.endswith(".tsv"):
            return pd.read_csv(uploaded, sep="\t")
        return pd.read_csv(uploaded)
    if sheet_url:
        return pd.read_csv(sheet_url)
    return pd.read_csv(StringIO(SAMPLE_TSV), sep="\t")

# ------------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def prepare_df(df):
    df = df.copy()
    df = df.sort_values("seconds_elapsed").reset_index(drop=True)

    # Distance
    df["lat_prev"] = df["latitude"].shift(1)
    df["lon_prev"] = df["longitude"].shift(1)
    df["dt"] = df["seconds_elapsed"].diff()
    df["dist_m"] = haversine(df["lat_prev"], df["lon_prev"],
                              df["latitude"], df["longitude"]).fillna(0)

    # Speed & acceleration
    df["speed_calc"] = (df["dist_m"] / df["dt"]).replace([np.inf, -np.inf], 0)
    df["accel"] = df["speed_calc"].diff().fillna(0)

    # Stop detection
    df["is_stop"] = df["speed_calc"] < 0.5

    return df

# ------------------------------------------------------------------
# UNSUPERVISED ML
# ------------------------------------------------------------------
def apply_dbscan_stops(df):
    stops = df[df["is_stop"]][["latitude", "longitude"]]
    if len(stops) < 5:
        df["stop_cluster"] = -1
        return df

    coords = np.radians(stops.values)
    db = DBSCAN(eps=30/6371000, min_samples=5, metric="haversine")
    labels = db.fit_predict(coords)

    df["stop_cluster"] = -1
    df.loc[stops.index, "stop_cluster"] = labels
    return df

def apply_kmeans_motion(df):
    X = df[["speed_calc", "accel"]].fillna(0)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["motion_cluster"] = kmeans.fit_predict(X)
    return df

def apply_isolation_forest(df):
    X = df[["speed_calc", "accel", "dist_m"]].fillna(0)
    iso = IsolationForest(contamination=0.02, random_state=42)
    df["anomaly"] = iso.fit_predict(X)
    return df

# ------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------
st.title("🗺️ GPS Analytics + Unsupervised ML (Industry Style)")

with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload CSV / TSV", type=["csv", "tsv"])
    sheet_url = st.text_input("Google Sheet CSV link")

df = load_data(uploaded, sheet_url)
df = prepare_df(df)
df = apply_dbscan_stops(df)
df = apply_kmeans_motion(df)
df = apply_isolation_forest(df)

# ------------------------------------------------------------------
# KPIs
# ------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Points", len(df))
c2.metric("Distance (km)", f"{df['dist_m'].sum()/1000:.2f}")
c3.metric("Avg Speed", f"{df['speed_calc'].mean():.2f} m/s")
c4.metric("Stops %", f"{df['is_stop'].mean()*100:.1f}%")
c5.metric("Anomalies", int((df["anomaly"]==-1).sum()))

# ------------------------------------------------------------------
# MAP (reuse your working MapLibre HTML)
# ------------------------------------------------------------------
coords = df[["longitude", "latitude"]].values.tolist()
payload = json.dumps({
    "coords": coords,
    "center": coords[0]
})

html = """
<!DOCTYPE html>
<html>
<head>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet"/>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<style>html,body,#map{height:100%;margin:0}</style>
</head>
<body>
<div id="map"></div>
<script>
const DATA = __DATA__;
const map = new maplibregl.Map({
  container:'map',
  style:{
    version:8,
    sources:{osm:{type:'raster',tiles:['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],tileSize:256}},
    layers:[{id:'osm',type:'raster',source:'osm'}]
  },
  center: DATA.center,
  zoom: 16
});
new maplibregl.Marker({color:'red'}).setLngLat(DATA.center).addTo(map);
</script>
</body>
</html>
"""
html = html.replace("__DATA__", payload)
components.html(html, height=500)

# ------------------------------------------------------------------
# ANALYTICS TABLES
# ------------------------------------------------------------------
st.subheader("📊 Unsupervised ML Insights")

st.markdown("### 🚦 Stop Clusters (DBSCAN)")
st.dataframe(
    df[df["stop_cluster"] >= 0][["latitude", "longitude", "stop_cluster"]]
    .groupby("stop_cluster").size().reset_index(name="count")
)

st.markdown("### 🚗 Movement Behavior (KMeans)")
st.dataframe(
    df.groupby("motion_cluster")[["speed_calc", "accel"]].mean()
)

st.markdown("### 🚨 Anomalous Events (Isolation Forest)")
st.dataframe(
    df[df["anomaly"] == -1][
        ["seconds_elapsed", "latitude", "longitude", "speed_calc", "dist_m"]
    ].head(20)
)

st.download_button(
    "⬇ Download Enriched Dataset",
    df.to_csv(index=False),
    "gps_ml_enriched.csv",
    "text/csv"
)
