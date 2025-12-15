import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
from io import StringIO

st.set_page_config(page_title="GPS Route + Data Science Analytics", layout="wide")

SAMPLE_TSV = """time\tseconds_elapsed\tbearingAccuracy\tspeedAccuracy\tverticalAccuracy\thorizontalAccuracy\tspeed\tbearing\taltitude\tlongitude\tlatitude
1.74296E+18\t0.255999756\t45\t1.5\t1.307455301\t20.59600067\t0\t0\t374.8000183\t78.5795227\t21.2710244
1.74296E+18\t3.175305176\t56.90444946\t0.587098598\t1.336648345\t12.78999996\t0.530843079\t102.8688431\t374.8000183\t78.5796365\t21.2710753
1.74296E+18\t3.242290771\t71.24659729\t0.923150122\t1.337318182\t10.19900036\t0.546774924\t102.1160355\t374.8000183\t78.5796392\t21.2710753
1.74296E+18\t4.241882813\t89.88391113\t0.781779766\t1.347314119\t8.765999794\t0.213110328\t88.08830261\t374.8000183\t78.5796205\t21.2710833
"""

# ---------- Core geo/math helpers ----------
def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points."""
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def robust_zscore(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad

def make_arrow_div_icon(bearing_deg: float) -> folium.DivIcon:
    html = f"""
    <div style="
        transform: rotate({bearing_deg}deg);
        font-size: 28px;
        color: #e11d48;
        text-shadow: 0 0 2px #000;
        line-height: 28px;
    ">▲</div>
    """
    return folium.DivIcon(html=html)

# ---------- Data loading / prep ----------
def load_data(sheet_csv_url: str | None, uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        # Support CSV + TSV
        name = uploaded_file.name.lower()
        if name.endswith(".tsv"):
            df = pd.read_csv(uploaded_file, sep="\t")
        else:
            df = pd.read_csv(uploaded_file)
        return df

    if sheet_csv_url and sheet_csv_url.strip():
        return pd.read_csv(sheet_csv_url)

    return pd.read_csv(StringIO(SAMPLE_TSV), sep="\t")

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    # Try to normalize column names
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    required = ["seconds_elapsed", "longitude", "latitude", "bearing"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    numeric_cols = ["seconds_elapsed", "longitude", "latitude", "bearing", "speed",
                    "horizontalAccuracy", "verticalAccuracy", "speedAccuracy", "bearingAccuracy", "altitude"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["seconds_elapsed", "longitude", "latitude", "bearing"])
    df = df.sort_values("seconds_elapsed").reset_index(drop=True)

    # Feature engineering
    df["lat_prev"] = df["latitude"].shift(1)
    df["lon_prev"] = df["longitude"].shift(1)
    df["t_prev"] = df["seconds_elapsed"].shift(1)

    df["dt"] = df["seconds_elapsed"] - df["t_prev"]
    df["dist_m"] = haversine_m(df["lat_prev"], df["lon_prev"], df["latitude"], df["longitude"])
    df.loc[0, "dist_m"] = 0.0
    df.loc[0, "dt"] = np.nan

    df["speed_mps_from_gps"] = df["dist_m"] / df["dt"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # If speed column missing or mostly NaN, use gps-derived
    if "speed" not in df.columns or df["speed"].isna().mean() > 0.5:
        df["speed"] = df["speed_mps_from_gps"]

    # Acceleration (m/s^2)
    df["accel_mps2"] = (df["speed"] - df["speed"].shift(1)) / df["dt"]

    # Turning angle based on bearing change
    df["bearing_change"] = (df["bearing"] - df["bearing"].shift(1))
    df["bearing_change"] = (df["bearing_change"] + 180) % 360 - 180  # wrap to [-180,180]

    # Stop detection (simple)
    df["is_stop"] = (df["speed"].fillna(0) < 0.5)

    # Anomaly detection
    df["z_speed_jump"] = robust_zscore(df["accel_mps2"].fillna(0))
    df["z_gps_jump"] = robust_zscore(df["dist_m"].fillna(0))
    df["is_anomaly"] = (np.abs(df["z_speed_jump"]) > 3.5) | (np.abs(df["z_gps_jump"]) > 3.5)

    return df

# ---------- Map builder ----------
def build_map(df: pd.DataFrame, t: float, show_anomalies=True) -> folium.Map:
    covered = df[df["seconds_elapsed"] <= t].copy()
    if covered.empty:
        covered = df.iloc[[0]].copy()

    idx = (df["seconds_elapsed"] - t).abs().idxmin()
    cur = df.loc[idx]

    center = [float(cur["latitude"]), float(cur["longitude"])]
    m = folium.Map(location=center, zoom_start=18, tiles="OpenStreetMap", control_scale=True)

    # Covered polyline
    route = covered[["latitude", "longitude"]].values.tolist()
    if len(route) >= 2:
        folium.PolyLine(route, weight=6, opacity=0.9).add_to(m)

    # Start marker
    start = df.iloc[0]
    folium.CircleMarker(
        location=[float(start["latitude"]), float(start["longitude"])],
        radius=5,
        popup="Start",
        fill=True
    ).add_to(m)

    # Anomaly markers (red dots)
    if show_anomalies and "is_anomaly" in df.columns:
        anoms = df[df["is_anomaly"]]
        for _, r in anoms.iterrows():
            folium.CircleMarker(
                location=[float(r["latitude"]), float(r["longitude"])],
                radius=4,
                color="red",
                fill=True,
                fill_opacity=0.8,
                popup=f"Anomaly<br>t={r['seconds_elapsed']:.2f}<br>dist_m={r['dist_m']:.1f}<br>accel={r['accel_mps2']:.2f}"
            ).add_to(m)

    # Moving arrow
    popup = folium.Popup(
        f"t={cur['seconds_elapsed']:.2f}s<br>"
        f"speed={float(cur.get('speed', 0) or 0):.3f} m/s<br>"
        f"bearing={cur['bearing']:.1f}°<br>"
        f"dist(step)={float(cur.get('dist_m', 0) or 0):.2f} m<br>"
        f"accel={float(cur.get('accel_mps2', 0) or 0):.2f} m/s²<br>"
        f"anomaly={bool(cur.get('is_anomaly', False))}",
        max_width=260
    )

    folium.Marker(
        location=center,
        icon=make_arrow_div_icon(float(cur["bearing"])),
        popup=popup
    ).add_to(m)

    return m

# ---------- UI ----------
st.title("🧠 GPS Route Playback + Crazy Data Science Analytics")

with st.sidebar:
    st.subheader("Load Data")
    uploaded = st.file_uploader("Upload CSV/TSV from computer", type=["csv", "tsv"])
    sheet_url = st.text_input(
        "Or Google Sheet CSV export link",
        placeholder="https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=0"
    )
    st.caption("If both are empty, sample data is used.")

try:
    raw = load_data(sheet_url, uploaded)
    df = prepare_df(raw)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

if df.empty:
    st.warning("No valid rows after cleaning.")
    st.stop()

t_min = float(df["seconds_elapsed"].min())
t_max = float(df["seconds_elapsed"].max())

if "playing" not in st.session_state:
    st.session_state.playing = False
if "t" not in st.session_state:
    st.session_state.t = t_min

with st.sidebar:
    st.subheader("Playback")
    c1, c2 = st.columns(2)
    if c1.button("▶ Play"):
        st.session_state.playing = True
    if c2.button("⏸ Pause"):
        st.session_state.playing = False

    speed_mult = st.slider("Speed (x)", 0.25, 10.0, 1.0, 0.25)
    step = st.slider("Step per tick (sec)", 0.05, 2.0, 0.20, 0.05)
    show_anomalies = st.checkbox("Show anomaly points", value=True)

    st.session_state.t = st.slider(
        "Current seconds_elapsed",
        min_value=t_min,
        max_value=t_max,
        value=float(st.session_state.t),
        step=0.01
    )

if st.session_state.playing:
    st_autorefresh(interval=250, key="route_refresh")
    st.session_state.t = min(t_max, st.session_state.t + step * speed_mult)

# KPI Section
total_distance_m = float(df["dist_m"].fillna(0).sum())
avg_speed = float(df["speed"].mean())
max_speed = float(df["speed"].max())
stop_ratio = float(df["is_stop"].mean() * 100)
anoms = int(df["is_anomaly"].sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Distance", f"{total_distance_m/1000:.3f} km")
k2.metric("Avg Speed", f"{avg_speed:.3f} m/s")
k3.metric("Max Speed", f"{max_speed:.3f} m/s")
k4.metric("Stop Points", f"{stop_ratio:.1f}%")
k5.metric("Anomalies", f"{anoms}")

# Layout
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("🗺️ Map Playback")
    m = build_map(df, st.session_state.t, show_anomalies=show_anomalies)
    st_folium(m, height=600, width=None)

with right:
    st.subheader("📌 Current Point")
    idx = (df["seconds_elapsed"] - st.session_state.t).abs().idxmin()
    cur = df.loc[idx]
    st.write(f"**t:** {float(cur['seconds_elapsed']):.2f}s")
    st.write(f"**speed:** {float(cur['speed']):.4f} m/s")
    st.write(f"**bearing:** {float(cur['bearing']):.1f}°")
    st.write(f"**step distance:** {float(cur.get('dist_m', 0) or 0):.2f} m")
    st.write(f"**accel:** {float(cur.get('accel_mps2', 0) or 0):.2f} m/s²")
    st.write(f"**anomaly:** {bool(cur.get('is_anomaly', False))}")

    st.subheader("⬇️ Download Enriched Data")
    out = df.drop(columns=["lat_prev", "lon_prev", "t_prev"], errors="ignore")
    st.download_button(
        "Download CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="gps_enriched.csv",
        mime="text/csv"
    )

st.divider()
st.subheader("📊 Analytics (Make students go: ‘WHAT!’)")

tab1, tab2, tab3, tab4 = st.tabs(["Speed", "Movement vs Stop", "Turning", "Anomalies"])

with tab1:
    fig = px.line(df, x="seconds_elapsed", y="speed", title="Speed vs Time (m/s)")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(df, x="speed", nbins=30, title="Speed Distribution")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    # moving vs stop
    mv = pd.DataFrame({
        "state": ["Moving", "Stopped"],
        "count": [int((~df["is_stop"]).sum()), int(df["is_stop"].sum())]
    })
    fig = px.pie(mv, names="state", values="count", title="Moving vs Stopped")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.line(df, x="seconds_elapsed", y="bearing_change", title="Bearing Change (Turning) vs Time (degrees)")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    an_df = df[df["is_anomaly"]].copy()
    st.write("Anomalies are detected using robust z-scores on **GPS jump** and **speed/acceleration jump**.")
    fig = px.scatter(
        df, x="seconds_elapsed", y="dist_m",
        color="is_anomaly",
        title="Step Distance vs Time (Anomaly points highlighted)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(an_df[["seconds_elapsed", "dist_m", "speed", "accel_mps2", "z_gps_jump", "z_speed_jump",
                        "latitude", "longitude"]].head(50), use_container_width=True)

st.subheader("🔎 Data Quality")
dq1, dq2, dq3 = st.columns(3)
dq1.metric("Rows", len(df))
dq2.metric("Missing values", int(df.isna().sum().sum()))
dq3.metric("Duplicate rows", int(df.duplicated().sum()))

st.dataframe(df.head(25), use_container_width=True)
