import streamlit as st
import pandas as pd
from io import StringIO
import folium
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Route Playback (Folium)", layout="wide")

SAMPLE_TSV = """time\tseconds_elapsed\tbearingAccuracy\tspeedAccuracy\tverticalAccuracy\thorizontalAccuracy\tspeed\tbearing\taltitude\tlongitude\tlatitude
1.74296E+18\t0.255999756\t45\t1.5\t1.307455301\t20.59600067\t0\t0\t374.8000183\t78.5795227\t21.2710244
1.74296E+18\t3.175305176\t56.90444946\t0.587098598\t1.336648345\t12.78999996\t0.530843079\t102.8688431\t374.8000183\t78.5796365\t21.2710753
1.74296E+18\t3.242290771\t71.24659729\t0.923150122\t1.337318182\t10.19900036\t0.546774924\t102.1160355\t374.8000183\t78.5796392\t21.2710753
1.74296E+18\t4.241882813\t89.88391113\t0.781779766\t1.347314119\t8.765999794\t0.213110328\t88.08830261\t374.8000183\t78.5796205\t21.2710833
"""

def load_df(sheet_csv_url: str | None) -> pd.DataFrame:
    if sheet_csv_url and sheet_csv_url.strip():
        df = pd.read_csv(sheet_csv_url)
    else:
        df = pd.read_csv(StringIO(SAMPLE_TSV), sep="\t")
    return df

def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["seconds_elapsed", "bearing", "longitude", "latitude"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    for c in ["seconds_elapsed", "bearing", "longitude", "latitude", "speed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["seconds_elapsed", "longitude", "latitude", "bearing"]).sort_values("seconds_elapsed")
    df = df.reset_index(drop=True)
    return df

def make_arrow_div_icon(bearing_deg: float) -> folium.DivIcon:
    # Rotated arrow using HTML/CSS. Arrow points up by default then rotated.
    # bearing: degrees clockwise from North (typical GPS bearing)
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

def build_map(df: pd.DataFrame, t: float) -> folium.Map:
    # Covered path
    covered = df[df["seconds_elapsed"] <= t].copy()
    if covered.empty:
        covered = df.iloc[[0]].copy()

    # Current point = nearest time
    idx = (df["seconds_elapsed"] - t).abs().idxmin()
    cur = df.loc[idx]

    center = [float(cur["latitude"]), float(cur["longitude"])]
    m = folium.Map(location=center, zoom_start=18, tiles="OpenStreetMap", control_scale=True)

    # Draw covered polyline
    route = covered[["latitude", "longitude"]].values.tolist()
    if len(route) >= 2:
        folium.PolyLine(route, weight=6, opacity=0.9).add_to(m)

    # Start & end markers (optional)
    start = df.iloc[0]
    folium.CircleMarker(
        location=[float(start["latitude"]), float(start["longitude"])],
        radius=5,
        popup="Start",
        fill=True
    ).add_to(m)

    # Moving arrow marker
    popup = folium.Popup(
        f"t={cur['seconds_elapsed']:.2f}s<br>"
        f"speed={float(cur.get('speed', 0) or 0):.3f}<br>"
        f"bearing={cur['bearing']:.1f}<br>"
        f"lat={cur['latitude']:.6f}<br>lon={cur['longitude']:.6f}",
        max_width=250
    )

    folium.Marker(
        location=center,
        icon=make_arrow_div_icon(float(cur["bearing"])),
        popup=popup
    ).add_to(m)

    return m


# ---------------- UI ----------------
st.title("🗺️ Route Playback (Folium + Moving Arrow)")

with st.sidebar:
    st.subheader("Data Source")
    sheet_url = st.text_input(
        "Google Sheet CSV export link",
        placeholder="https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=0"
    )
    st.caption("If left empty, the app uses the sample data you provided.")

try:
    df = prep_df(load_df(sheet_url))
except Exception as e:
    st.error(f"Failed to load/prepare data: {e}")
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

    st.session_state.t = st.slider(
        "Current seconds_elapsed",
        min_value=t_min,
        max_value=t_max,
        value=float(st.session_state.t),
        step=0.01
    )

# Auto refresh when playing (no crashing attributes)
if st.session_state.playing:
    st_autorefresh(interval=250, key="route_refresh")
    st.session_state.t = min(t_max, st.session_state.t + step * speed_mult)

m = build_map(df, st.session_state.t)

left, right = st.columns([2, 1], gap="large")
with left:
    st_folium(m, width=None, height=600)

with right:
    st.subheader("Current Values")
    idx = (df["seconds_elapsed"] - st.session_state.t).abs().idxmin()
    cur = df.loc[idx]
    st.write(f"**seconds_elapsed:** {float(cur['seconds_elapsed']):.2f}")
    st.write(f"**speed:** {float(cur.get('speed', 0) or 0):.4f}")
    st.write(f"**bearing:** {float(cur['bearing']):.1f}")
    st.write(f"**lat:** {float(cur['latitude']):.6f}")
    st.write(f"**lon:** {float(cur['longitude']):.6f}")

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
