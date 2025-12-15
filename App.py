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

    for col in ["seconds_elapsed", "latitude", "longitude", "speed", "bearing"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Required
    for col in ["seconds_elapsed", "latitude", "longitude"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=["seconds_elapsed", "latitude", "longitude"]).sort_values("seconds_elapsed")
    df = df.reset_index(drop=True)

    # Distance engineering
    R = 6371000.0
    lat1 = np.radians(df["latitude"].shift(1))
    lon1 = np.radians(df["longitude"].shift(1))
    lat2 = np.radians(df["latitude"])
    lon2 = np.radians(df["longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    dist = 2 * R * np.arcsin(np.sqrt(a))
    df["dist_m"] = dist.fillna(0)

    df["dt"] = df["seconds_elapsed"].diff()
    if "speed" not in df.columns or df["speed"].isna().mean() > 0.5:
        df["speed"] = (df["dist_m"] / df["dt"]).replace([np.inf, -np.inf], np.nan)

    df["is_stop"] = df["speed"].fillna(0) < 0.5

    mu, sd = df["dist_m"].mean(), df["dist_m"].std(ddof=0)
    df["anomaly"] = df["dist_m"] > (mu + 3 * (sd if sd > 0 else 1))
    return df

def build_timeline_map(df: pd.DataFrame, period_s: float, transition_ms: int, show_anoms: bool):
    center = [float(df.loc[0, "latitude"]), float(df.loc[0, "longitude"])]

    # Use a tile provider that often works reliably
    m = folium.Map(location=center, zoom_start=18, tiles="CartoDB positron", control_scale=True)

    # Static route
    route = df[["latitude", "longitude"]].values.tolist()
    if len(route) >= 2:
        folium.PolyLine(route, weight=6, opacity=0.7).add_to(m)

    # Static anomaly points
    if show_anoms:
        for _, r in df[df["anomaly"]].iterrows():
            folium.CircleMarker(
                location=[float(r["latitude"]), float(r["longitude"])],
                radius=4,
                color="red",
                fill=True,
                fill_opacity=0.85,
                popup=f"Anomaly | t={r['seconds_elapsed']:.2f}s | dist={r['dist_m']:.2f}m"
            ).add_to(m)

    # Build timeline points
    base = datetime.now(timezone.utc).replace(microsecond=0)
    times = [base + timedelta(seconds=float(s)) for s in df["seconds_elapsed"].values]

    features = []
    for i, r in df.iterrows():
        t = times[i].isoformat()
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r["longitude"]), float(r["latitude"])]},
            "properties": {
                "time": t,
                "popup": (
                    f"t={float(r['seconds_elapsed']):.2f}s<br>"
                    f"speed={float(r.get('speed', 0) or 0):.3f} m/s<br>"
                    f"dist(step)={float(r.get('dist_m', 0) or 0):.2f} m<br>"
                    f"stop={bool(r.get('is_stop', False))}<br>"
                    f"anomaly={bool(r.get('anomaly', False))}"
                ),
                "icon": "circle",
                "iconstyle": {"fillColor": "#e11d48", "fillOpacity": 0.95, "stroke": "true", "radius": 7}
            }
        })

    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period=f"PT{max(period_s, 0.05)}S",
        add_last_point=True,
        auto_play=True,
        loop=False,
        max_speed=8,
        loop_button=True,
        time_slider_drag_update=True,
        transition_time=transition_ms
    ).add_to(m)

    return m

# ---------------- UI ----------------
st.title("🗺️ Smooth GPS Playback (No flashing, stable render)")

with st.sidebar:
    uploaded = st.file_uploader("Upload CSV/TSV", type=["csv", "tsv"])
    sheet_url = st.text_input("Google Sheet CSV export link")
    period_s = st.slider("Frame period (seconds)", 0.05, 2.0, 0.20, 0.05)
    transition_ms = st.slider("Transition smoothness (ms)", 0, 1500, 400, 50)
    show_anoms = st.checkbox("Show anomaly points", value=True)

try:
    raw = load_data(uploaded, sheet_url)
    df = prepare_df(raw)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

if df.empty:
    st.warning("No valid rows after cleaning.")
    st.stop()

# KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", len(df))
c2.metric("Total Distance", f"{df['dist_m'].sum()/1000:.3f} km")
c3.metric("Avg Speed", f"{df['speed'].mean():.3f} m/s")
c4.metric("Stops", f"{df['is_stop'].mean()*100:.1f}%")
c5.metric("Anomalies", int(df["anomaly"].sum()))

m = build_timeline_map(df, period_s=period_s, transition_ms=transition_ms, show_anoms=show_anoms)

# Render map WITHOUT st_folium (prevents “no map” issues in some environments)
components.html(m._repr_html_(), height=700)

st.subheader("Analytics (no Plotly)")
st.line_chart(df.set_index("seconds_elapsed")["speed"], height=250)
st.bar_chart(df["dist_m"], height=250)

st.download_button(
    "Download enriched CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="gps_enriched.csv",
    mime="text/csv"
)
