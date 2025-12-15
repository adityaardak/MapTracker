import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import TimestampedGeoJson
from streamlit_folium import st_folium
from io import StringIO
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="Smooth GPS Playback", layout="wide")

SAMPLE_TSV = """time\tseconds_elapsed\tbearingAccuracy\tspeedAccuracy\tverticalAccuracy\thorizontalAccuracy\tspeed\tbearing\taltitude\tlongitude\tlatitude
1.74296E+18\t0.255999756\t45\t1.5\t1.307455301\t20.59600067\t0\t0\t374.8000183\t78.5795227\t21.2710244
1.74296E+18\t3.175305176\t56.90444946\t0.587098598\t1.336648345\t12.78999996\t0.530843079\t102.8688431\t374.8000183\t78.5796365\t21.2710753
1.74296E+18\t3.242290771\t71.24659729\t0.923150122\t1.337318182\t10.19900036\t0.546774924\t102.1160355\t374.8000183\t78.5796392\t21.2710753
1.74296E+18\t4.241882813\t89.88391113\t0.781779766\t1.347314119\t8.765999794\t0.213110328\t88.08830261\t374.8000183\t78.5796205\t21.2710833
"""

def load_data(uploaded, sheet_url: str | None):
    if uploaded is not None:
        name = uploaded.name.lower()
        if name.endswith(".tsv"):
            return pd.read_csv(uploaded, sep="\t")
        return pd.read_csv(uploaded)
    if sheet_url and sheet_url.strip():
        return pd.read_csv(sheet_url.strip())
    return pd.read_csv(StringIO(SAMPLE_TSV), sep="\t")

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    required = ["seconds_elapsed", "latitude", "longitude"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    for c in ["seconds_elapsed", "latitude", "longitude", "speed", "bearing"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["seconds_elapsed", "latitude", "longitude"]).sort_values("seconds_elapsed")
    df = df.reset_index(drop=True)

    # Simple engineered features (for “DS wow”)
    df["lat_prev"] = df["latitude"].shift(1)
    df["lon_prev"] = df["longitude"].shift(1)
    df["dt"] = df["seconds_elapsed"].diff()

    # Haversine step distance (m)
    R = 6371000.0
    lat1 = np.radians(df["lat_prev"])
    lon1 = np.radians(df["lon_prev"])
    lat2 = np.radians(df["latitude"])
    lon2 = np.radians(df["longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    df["dist_m"] = (2 * R * np.arcsin(np.sqrt(a))).fillna(0)

    # GPS-derived speed if speed missing
    if "speed" not in df.columns or df["speed"].isna().mean() > 0.5:
        df["speed"] = (df["dist_m"] / df["dt"]).replace([np.inf, -np.inf], np.nan)

    df["is_stop"] = df["speed"].fillna(0) < 0.5

    # Simple anomaly flag: big jumps
    mu, sd = df["dist_m"].mean(), df["dist_m"].std(ddof=0)
    df["anomaly"] = df["dist_m"] > (mu + 3 * (sd if sd > 0 else 1))

    return df

def build_smooth_timeline_map(df: pd.DataFrame, point_period_s: float, transition_ms: int):
    # Center
    center = [float(df.loc[0, "latitude"]), float(df.loc[0, "longitude"])]
    m = folium.Map(location=center, zoom_start=18, tiles="OpenStreetMap", control_scale=True)

    # Full route polyline (static)
    route = df[["latitude", "longitude"]].values.tolist()
    if len(route) >= 2:
        folium.PolyLine(route, weight=6, opacity=0.7).add_to(m)

    # Optional anomaly markers (static)
    anoms = df[df["anomaly"]]
    for _, r in anoms.iterrows():
        folium.CircleMarker(
            location=[float(r["latitude"]), float(r["longitude"])],
            radius=4,
            color="red",
            fill=True,
            fill_opacity=0.85,
            popup=f"Anomaly<br>t={r['seconds_elapsed']:.2f}s<br>dist={r['dist_m']:.2f}m"
        ).add_to(m)

    # Create a “fake real time” datetime sequence from seconds_elapsed
    # Leaflet timeline wants ISO timestamps
    base = datetime.now(timezone.utc).replace(microsecond=0)
    times = [base + timedelta(seconds=float(s)) for s in df["seconds_elapsed"].values]

    features = []
    for i, r in df.iterrows():
        t = times[i].isoformat()
        popup = (
            f"t={float(r['seconds_elapsed']):.2f}s<br>"
            f"speed={float(r.get('speed', 0) or 0):.3f} m/s<br>"
            f"dist(step)={float(r.get('dist_m', 0) or 0):.2f} m<br>"
            f"stop={bool(r.get('is_stop', False))}<br>"
            f"anomaly={bool(r.get('anomaly', False))}"
        )
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r["longitude"]), float(r["latitude"])]},
            "properties": {
                "time": t,
                "popup": popup,
                "icon": "circle",
                "iconstyle": {
                    "fillColor": "#e11d48",
                    "fillOpacity": 0.95,
                    "stroke": "true",
                    "radius": 7
                }
            }
        })

    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period=f"PT{max(point_period_s, 0.05)}S",
        add_last_point=True,           # moving dot stays visible
        auto_play=True,                # starts playing
        loop=False,
        max_speed=5,                   # user can speed up in UI
        loop_button=True,
        date_options="HH:mm:ss",
        time_slider_drag_update=True,
        transition_time=transition_ms  # smoother transitions
    ).add_to(m)

    return m

# -------------------- UI --------------------
st.title("🗺️ Smooth GPS Playback (No flashing)")

with st.sidebar:
    st.subheader("Data Source")
    uploaded = st.file_uploader("Upload CSV / TSV", type=["csv", "tsv"])
    sheet_url = st.text_input("Google Sheet CSV export link", placeholder="https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=0")

    st.subheader("Smooth Playback Settings")
    # This controls how dense the time axis is; for very dense points, increase a bit
    point_period_s = st.slider("Frame period (seconds)", 0.05, 2.0, 0.20, 0.05)
    transition_ms = st.slider("Transition smoothness (ms)", 0, 1500, 400, 50)

try:
    raw = load_data(uploaded, sheet_url)
    df = prepare_df(raw)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

if df.empty:
    st.warning("No valid rows to display.")
    st.stop()

# KPIs (analytics “wow”)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows", f"{len(df)}")
k2.metric("Total Distance", f"{df['dist_m'].sum()/1000:.3f} km")
k3.metric("Avg Speed", f"{df['speed'].mean():.3f} m/s")
k4.metric("Stop %", f"{(df['is_stop'].mean()*100):.1f}%")
k5.metric("Anomalies", f"{int(df['anomaly'].sum())}")

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("✅ Smooth Map Animation (client-side, no Streamlit reruns)")
    m = build_smooth_timeline_map(df, point_period_s=point_period_s, transition_ms=transition_ms)
    st_folium(m, height=650, width=None)

with right:
    st.subheader("Data Science Panel")
    st.write("Speed vs Time")
    if "speed" in df.columns:
        st.line_chart(df.set_index("seconds_elapsed")["speed"])

    st.write("Step Distance (m)")
    st.bar_chart(df["dist_m"])

    st.write("Anomaly table (top)")
    st.dataframe(df[df["anomaly"]][["seconds_elapsed", "dist_m", "speed", "latitude", "longitude"]].head(30),
                 use_container_width=True)

    st.download_button(
        "Download enriched CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="gps_enriched.csv",
        mime="text/csv"
    )
