import streamlit as st
import pandas as pd
import numpy as np
import folium
from io import StringIO
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="GPS Route + Data Science Analytics", layout="wide")

# ---------------- SAMPLE DATA ----------------
SAMPLE_TSV = """time\tseconds_elapsed\tbearingAccuracy\tspeedAccuracy\tverticalAccuracy\thorizontalAccuracy\tspeed\tbearing\taltitude\tlongitude\tlatitude
1.74296E+18\t0.255999756\t45\t1.5\t1.307455301\t20.59600067\t0\t0\t374.8000183\t78.5795227\t21.2710244
1.74296E+18\t3.175305176\t56.90444946\t0.587098598\t1.336648345\t12.78999996\t0.530843079\t102.8688431\t374.8000183\t78.5796365\t21.2710753
1.74296E+18\t3.242290771\t71.24659729\t0.923150122\t1.337318182\t10.19900036\t0.546774924\t102.1160355\t374.8000183\t78.5796392\t21.2710753
1.74296E+18\t4.241882813\t89.88391113\t0.781779766\t1.347314119\t8.765999794\t0.213110328\t88.08830261\t374.8000183\t78.5796205\t21.2710833
"""

# ---------------- HELPERS ----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def arrow_icon(bearing):
    html = f"""
    <div style="
        transform: rotate({bearing}deg);
        font-size:28px;
        color:red;
        text-shadow:1px 1px 2px black;
    ">▲</div>
    """
    return folium.DivIcon(html=html)

def load_data(uploaded, sheet_url):
    if uploaded:
        if uploaded.name.endswith(".tsv"):
            return pd.read_csv(uploaded, sep="\t")
        return pd.read_csv(uploaded)
    if sheet_url:
        return pd.read_csv(sheet_url)
    return pd.read_csv(StringIO(SAMPLE_TSV), sep="\t")

def prepare_data(df):
    for c in ["seconds_elapsed","latitude","longitude","bearing","speed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["seconds_elapsed","latitude","longitude","bearing"])
    df = df.sort_values("seconds_elapsed").reset_index(drop=True)

    df["lat_prev"] = df["latitude"].shift(1)
    df["lon_prev"] = df["longitude"].shift(1)
    df["t_prev"] = df["seconds_elapsed"].shift(1)

    df["dt"] = df["seconds_elapsed"] - df["t_prev"]
    df["dist_m"] = haversine(df["lat_prev"], df["lon_prev"], df["latitude"], df["longitude"])
    df.loc[0,["dt","dist_m"]] = 0

    df["speed_calc"] = df["dist_m"] / df["dt"].replace(0,np.nan)
    df["accel"] = (df["speed"] - df["speed"].shift(1)) / df["dt"]

    df["is_stop"] = df["speed"].fillna(0) < 0.5

    # Simple anomaly rule
    df["anomaly"] = (df["dist_m"] > df["dist_m"].mean() + 3*df["dist_m"].std())

    return df

def build_map(df, t):
    idx = (df["seconds_elapsed"] - t).abs().idxmin()
    cur = df.loc[idx]

    m = folium.Map(
        location=[cur["latitude"], cur["longitude"]],
        zoom_start=18,
        tiles="OpenStreetMap"
    )

    covered = df[df["seconds_elapsed"] <= t]
    folium.PolyLine(
        covered[["latitude","longitude"]].values.tolist(),
        weight=6
    ).add_to(m)

    # anomaly points
    for _,r in df[df["anomaly"]].iterrows():
        folium.CircleMarker(
            [r["latitude"], r["longitude"]],
            radius=4,
            color="red",
            fill=True,
            popup="Anomaly"
        ).add_to(m)

    folium.Marker(
        [cur["latitude"], cur["longitude"]],
        icon=arrow_icon(cur["bearing"]),
        popup=f"""
        t={cur['seconds_elapsed']:.2f}s<br>
        speed={cur['speed']:.2f} m/s<br>
        dist={cur['dist_m']:.2f} m
        """
    ).add_to(m)

    return m, cur

# ---------------- UI ----------------
st.title("🧠 GPS Route Playback + Data Science Analytics")

with st.sidebar:
    st.subheader("Load Data")
    uploaded = st.file_uploader("Upload CSV / TSV", type=["csv","tsv"])
    sheet_url = st.text_input("Google Sheet CSV link")

df = prepare_data(load_data(uploaded, sheet_url))

t_min, t_max = df["seconds_elapsed"].min(), df["seconds_elapsed"].max()

if "t" not in st.session_state:
    st.session_state.t = t_min
if "play" not in st.session_state:
    st.session_state.play = False

with st.sidebar:
    c1,c2 = st.columns(2)
    if c1.button("▶ Play"): st.session_state.play = True
    if c2.button("⏸ Pause"): st.session_state.play = False

    speed = st.slider("Playback speed",0.25,5.0,1.0)
    step = st.slider("Step (sec)",0.05,1.0,0.2)

    st.session_state.t = st.slider(
        "Current Time",
        float(t_min), float(t_max),
        float(st.session_state.t)
    )

if st.session_state.play:
    st_autorefresh(interval=300, key="refresh")
    st.session_state.t = min(t_max, st.session_state.t + step*speed)

# KPIs
k1,k2,k3,k4 = st.columns(4)
k1.metric("Distance (m)", f"{df['dist_m'].sum():.2f}")
k2.metric("Avg Speed", f"{df['speed'].mean():.2f}")
k3.metric("Max Speed", f"{df['speed'].max():.2f}")
k4.metric("Stops (%)", f"{df['is_stop'].mean()*100:.1f}")

# Layout
left,right = st.columns([2,1])

with left:
    st.subheader("🗺️ Route Playback")
    m, cur = build_map(df, st.session_state.t)
    st_folium(m, height=600)

with right:
    st.subheader("📍 Current Point")
    st.write(cur[["seconds_elapsed","speed","dist_m","accel","anomaly"]])

    st.subheader("⬇ Download Enriched CSV")
    st.download_button(
        "Download",
        df.to_csv(index=False),
        file_name="gps_enriched.csv"
    )

st.divider()
st.subheader("📊 Analytics (No Plotly)")

st.write("Speed vs Time")
st.line_chart(df.set_index("seconds_elapsed")["speed"])

st.write("Distance per Step")
st.bar_chart(df["dist_m"])

st.write("Acceleration vs Time")
st.line_chart(df.set_index("seconds_elapsed")["accel"])

st.write("Anomaly Points")
st.dataframe(df[df["anomaly"]][["seconds_elapsed","dist_m","speed","latitude","longitude"]])
