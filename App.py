import streamlit as st
import pandas as pd
import numpy as np
import folium
import streamlit.components.v1 as components
from folium.plugins import TimestampedGeoJson
from io import StringIO
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="Smooth GPS Playback", layout="wide")

SAMPLE_TSV = """time\tseconds_elapsed\tbearingAccuracy\tspeedAccuracy\tverticalAccuracy\thorizontalAccuracy\tspeed\tbearing\taltitude\tlongitude\tlatitude
1.74296E+18\t0.255999756\t45\t1.5\t1.307455301\t20.59600067\t0\t0\t374.8000183\t78.5795227\t21.2710244
1.74296E+18\t3.175305176\t56.90444946\t0.587098598\t1.336648345\t12.78999996\t0.530843079\t102.8688431\t374.8000183\t78.5796365\t21.2710753
1.74296E+18\t3.242290771\t71.24659729\t0.923150122\t1.337318182\t10.19900036\t0.546774924\t102.1160355\t374.8000183\t78.5796392\t21.2710753
1.74296E+18\t4.241882813\t89.88391113\t0.781779766\t1.347314119\t8.765999794\t0.213110328\t88.08830261\t374.8000183\t78.5796205\t21.2710833
"""

# ------------------------- Helpers -------------------------
def load_data(uploaded, sheet_url: str | None) -> pd.DataFrame:
    if uploaded is not None:
        name = uploaded.name.lower()
        if name.endswith(".tsv"):
            return pd.read_csv(uploaded, sep="\t")
        return pd.read_csv(uploaded)

    if sheet_url and sheet_url.strip():
        return pd.read_csv(sheet_url.strip())

    return pd.read_csv(StringIO(SAMPLE_TSV), sep="\t")

def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in meters."""
    R = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    required = ["seconds_elapsed", "latitude", "longitude"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # numeric
    for c in ["seconds_elapsed", "latitude", "longitude", "speed", "bearing"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["seconds_elapsed", "latitude", "longitude"]).sort_values("seconds_elapsed")
    df = df.reset_index(drop=True)

    # Engineering
    df["lat_prev"] = df["latitude"].shift(1)
    df["lon_prev"] = df["longitude"].shift(1)
    df["dt"] = df["seconds_elapsed"].diff()

    df["dist_m"] = haversine_m(df["lat_prev"], df["lon_prev"], df["latitude"], df["longitude"]).fillna(0)

    # If speed missing or mostly NaN, derive from GPS
    if "speed" not in df.columns or df["speed"].isna().mean() > 0.5:
        df["speed"] = (df["dist_m"] / df["dt"]).replace([np.inf, -np.inf], np.nan)

    df["accel_mps2"] = (df["speed"] - df["speed"].shift(1)) / df["dt"]
    df["is_stop"] = df["speed"].fillna(0) < 0.5

    # Simple anomaly: huge jumps in step distance
    mu = df["dist_m"].mean()
    sd = df["dist_m"].std(ddof=0)
    thr = mu + 3 * (sd if sd > 0 else 1.0)
    df["anomaly"] = df["dist_m"] > thr

    return df

def build_timeline_map(df: pd.DataFrame, period_s: float, transition_ms: int, show_anoms: bool):
    # Center
    center = [float(df.loc[0, "latitude"]), float(df.loc[0, "longitude"])]

    # Use a stable tile provider
    m = folium.Map(location=center, zoom_start=18, tiles="CartoDB positron", control_scale=True)

    # Background route polyline
    route_latlon = df[["latitude", "longitude"]].astype(float).values.tolist()
    if len(route_latlon) >= 2:
        folium.PolyLine(route_latlon, weight=6, opacity=0.35).add_to(m)

    # Optional anomaly markers (static)
    if show_anoms:
        anoms = df[df["anomaly"]]
        for _, r in anoms.iterrows():
            folium.CircleMarker(
                location=[float(r["latitude"]), float(r["longitude"])],
                radius=4,
                color="red",
                fill=True,
                fill_opacity=0.85,
                popup=f"Anomaly | t={r['seconds_elapsed']:.2f}s | dist={r['dist_m']:.2f}m"
            ).add_to(m)

    # Timeline timestamps (ISO8601)
    base = datetime.now(timezone.utc).replace(microsecond=0)
    times = [(base + timedelta(seconds=float(s))).isoformat() for s in df["seconds_elapsed"].astype(float).values]

    # IMPORTANT: GeoJSON LineString wants coords [lon, lat]
    coords_lonlat = df[["longitude", "latitude"]].astype(float).values.tolist()

    # One LineString feature + times list => smooth moving locator
    feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords_lonlat},
        "properties": {
            "times": times,  # must match length of coordinates
            "style": {"color": "#e11d48", "weight": 6, "opacity": 0.9},
        },
    }

    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": [feature]},
        period=f"PT{max(period_s, 0.05)}S",
        add_last_point=True,           # shows the moving locator
        auto_play=False,               # user clicks ▶ (reliable across browsers)
        loop=False,
        max_speed=12,
        loop_button=True,
        time_slider_drag_update=True,
        transition_time=int(transition_ms),
        date_options="HH:mm:ss",
    ).add_to(m)

    # On-map hint
    folium.Marker(
        center,
        icon=folium.DivIcon(
            html="<div style='font-size:12px;color:#111;background:#fff;padding:6px 8px;border-radius:8px;border:1px solid #ddd;'>Click ▶ (bottom-left) to play</div>"
        )
    ).add_to(m)

    return m

# ------------------------- UI -------------------------
st.title("🗺️ Smooth GPS Playback (No flashing) + Analytics")

with st.sidebar:
    st.subheader("Data Source")
    uploaded = st.file_uploader("Upload CSV / TSV", type=["csv", "tsv"])
    sheet_url = st.text_input("Google Sheet CSV export link", placeholder="https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=0")

    st.subheader("Playback Settings")
    period_s = st.slider("Frame period (seconds)", 0.05, 2.0, 0.20, 0.05)
    transition_ms = st.slider("Transition smoothness (ms)", 0, 1500, 450, 50)
    show_anoms = st.checkbox("Show anomaly points", value=True)

try:
    raw = load_data(uploaded, sheet_url)
    df = prepare_df(raw)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

if df.empty or len(df) < 2:
    st.warning("Need at least 2 valid rows (latitude, longitude, seconds_elapsed) to animate.")
    st.dataframe(df)
    st.stop()

# KPIs
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows", f"{len(df)}")
k2.metric("Total Distance", f"{df['dist_m'].sum()/1000:.3f} km")
k3.metric("Avg Speed", f"{df['speed'].mean():.3f} m/s")
k4.metric("Stops", f"{df['is_stop'].mean()*100:.1f}%")
k5.metric("Anomalies", f"{int(df['anomaly'].sum())}")

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("✅ Map + Smooth Locator (bottom-left ▶ controls)")
    m = build_timeline_map(df, period_s=period_s, transition_ms=transition_ms, show_anoms=show_anoms)
    # Render folium reliably
    components.html(m._repr_html_(), height=700)

with right:
    st.subheader("📊 Analytics (No Plotly)")

    st.write("Speed vs Time")
    if "speed" in df.columns:
        st.line_chart(df.set_index("seconds_elapsed")["speed"], height=220)

    st.write("Step Distance (m)")
    st.bar_chart(df["dist_m"], height=220)

    st.write("Acceleration vs Time")
    if "accel_mps2" in df.columns:
        accel_series = df.set_index("seconds_elapsed")["accel_mps2"].replace([np.inf, -np.inf], np.nan).fillna(0)
        st.line_chart(accel_series, height=220)

    st.write("Anomaly Points (table)")
    st.dataframe(
        df[df["anomaly"]][["seconds_elapsed", "dist_m", "speed", "latitude", "longitude"]].head(50),
        use_container_width=True
    )

    st.download_button(
        "⬇ Download enriched CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="gps_enriched.csv",
        mime="text/csv"
    )

st.divider()
st.subheader("Preview Data")
st.dataframe(df.head(50), use_container_width=True)
