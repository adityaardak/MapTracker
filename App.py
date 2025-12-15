import streamlit as st
import pandas as pd
import pydeck as pdk
from io import StringIO
from datetime import datetime, timezone
import base64

st.set_page_config(page_title="Route Playback Map", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
SAMPLE_CSV = """time\tseconds_elapsed\tbearingAccuracy\tspeedAccuracy\tverticalAccuracy\thorizontalAccuracy\tspeed\tbearing\taltitude\tlongitude\tlatitude
1.74296E+18\t0.255999756\t45\t1.5\t1.307455301\t20.59600067\t0\t0\t374.8000183\t78.5795227\t21.2710244
1.74296E+18\t3.175305176\t56.90444946\t0.587098598\t1.336648345\t12.78999996\t0.530843079\t102.8688431\t374.8000183\t78.5796365\t21.2710753
1.74296E+18\t3.242290771\t71.24659729\t0.923150122\t1.337318182\t10.19900036\t0.546774924\t102.1160355\t374.8000183\t78.5796392\t21.2710753
1.74296E+18\t4.241882813\t89.88391113\t0.781779766\t1.347314119\t8.765999794\t0.213110328\t88.08830261\t374.8000183\t78.5796205\t21.2710833
"""

def to_datetime_from_ns_like(x):
    """
    Your 'time' looks like ~1.74296E+18 which is typically nanoseconds since epoch.
    We'll parse it safely and convert to UTC datetime.
    """
    try:
        # handle scientific notation strings
        val = int(float(x))
        # if it's in nanoseconds, convert to seconds
        dt = datetime.fromtimestamp(val / 1_000_000_000, tz=timezone.utc)
        return dt
    except Exception:
        return None

def make_arrow_data_uri():
    # Small SVG arrow (triangle-ish) pointing "up" initially; we rotate via bearing.
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">
      <polygon points="32,4 56,60 32,48 8,60" fill="#FF3B30"/>
      <circle cx="32" cy="52" r="4" fill="#111111"/>
    </svg>
    """.strip()
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"

ARROW_ICON = make_arrow_data_uri()

def load_data_from_source(sheet_csv_url: str | None):
    if sheet_csv_url:
        df = pd.read_csv(sheet_csv_url)
    else:
        # sample is tab-separated
        df = pd.read_csv(StringIO(SAMPLE_CSV), sep="\t")
    return df

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleaning / types
    for col in ["longitude", "latitude", "seconds_elapsed", "bearing", "speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build a better time column
    if "time" in df.columns:
        df["time_dt_utc"] = df["time"].apply(to_datetime_from_ns_like)
    else:
        df["time_dt_utc"] = None

    # If seconds_elapsed missing, create a fallback index time
    if "seconds_elapsed" not in df.columns or df["seconds_elapsed"].isna().all():
        df["seconds_elapsed"] = range(len(df))

    df = df.dropna(subset=["longitude", "latitude", "seconds_elapsed"]).sort_values("seconds_elapsed")
    df = df.reset_index(drop=True)
    return df

def build_deck(df: pd.DataFrame, t: float):
    """
    t is "current time" in seconds_elapsed.
    We'll show the route covered up to t, plus an arrow at the closest point.
    """
    if df.empty:
        return None

    # Determine covered points
    covered = df[df["seconds_elapsed"] <= t].copy()
    if covered.empty:
        covered = df.iloc[[0]].copy()

    # Pick current point (nearest)
    idx = (df["seconds_elapsed"] - t).abs().idxmin()
    cur = df.loc[[idx]].copy()

    # Line path for covered route
    path_coords = covered[["longitude", "latitude"]].values.tolist()
    line_df = pd.DataFrame({"path": [path_coords]})

    # Arrow icon layer expects dict in "icon_data"
    cur["icon_data"] = cur.apply(
        lambda r: {
            "url": ARROW_ICON,
            "width": 64,
            "height": 64,
            "anchorY": 64,  # anchor bottom
        },
        axis=1,
    )

    # Deck.gl layers
    line_layer = pdk.Layer(
        "PathLayer",
        data=line_df,
        get_path="path",
        get_width=6,
        width_min_pixels=3,
        pickable=False,
    )

    # Optional: show all points faintly
    points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[longitude, latitude]",
        get_radius=3,
        radius_min_pixels=2,
        pickable=True,
    )

    # Moving arrow (rotation by bearing)
    icon_layer = pdk.Layer(
        "IconLayer",
        data=cur,
        get_icon="icon_data",
        get_position="[longitude, latitude]",
        get_size=5,
        size_scale=10,
        get_angle="bearing",   # rotate using bearing
        pickable=True,
    )

    # View state centered around current point
    view_state = pdk.ViewState(
        latitude=float(cur["latitude"].iloc[0]),
        longitude=float(cur["longitude"].iloc[0]),
        zoom=17,
        pitch=45,
        bearing=0,
    )

    tooltip = {
        "html": "<b>Seconds:</b> {seconds_elapsed}<br/>"
                "<b>Speed:</b> {speed}<br/>"
                "<b>Bearing:</b> {bearing}<br/>"
                "<b>Lat:</b> {latitude}<br/>"
                "<b>Lon:</b> {longitude}",
        "style": {"fontSize": "12px"},
    }

    deck = pdk.Deck(
        layers=[line_layer, points_layer, icon_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/streets-v11",  # works without token in many setups; otherwise still renders tiles via default
    )
    return deck, cur

# -----------------------------
# UI
# -----------------------------
st.title("📍 Route Playback Map (Moving Arrow)")

with st.sidebar:
    st.subheader("Data Source")
    st.write("Use a Google Sheet **CSV export link** (recommended) or run with sample data.")
    sheet_url = st.text_input(
        "Google Sheet CSV URL",
        placeholder="https://docs.google.com/spreadsheets/d/.../export?format=csv&gid=0",
    )
    st.caption(
        "Tip: In Google Sheets → File → Share/Publish (or set access) → use an export CSV link."
    )

    st.subheader("Playback Controls")

# Load + prep
try:
    raw = load_data_from_source(sheet_url if sheet_url.strip() else None)
    df = prepare_df(raw)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if df.empty:
    st.warning("No valid rows found. Ensure columns include: longitude, latitude, seconds_elapsed, bearing.")
    st.stop()

# Playback range
t_min = float(df["seconds_elapsed"].min())
t_max = float(df["seconds_elapsed"].max())

# Session playback state
if "playing" not in st.session_state:
    st.session_state.playing = False
if "t" not in st.session_state:
    st.session_state.t = t_min

with st.sidebar:
    col1, col2 = st.columns(2)
    if col1.button("▶ Play"):
        st.session_state.playing = True
    if col2.button("⏸ Pause"):
        st.session_state.playing = False

    speed_mult = st.slider("Playback speed (x)", 0.25, 10.0, 1.0, 0.25)
    step_sec = st.slider("Step per tick (seconds)", 0.05, 2.0, 0.20, 0.05)

    # Manual scrub
    st.session_state.t = st.slider(
        "Current time (seconds_elapsed)",
        min_value=t_min,
        max_value=t_max,
        value=float(st.session_state.t),
        step=0.01,
    )

# Auto-advance if playing
if st.session_state.playing:
    # autorefresh tick
    st_autorefresh = st.experimental_data_editor  # dummy to keep lint quiet (Streamlit older/newer)
    st.experimental_rerun  # reference
    # Use built-in autorefresh
    count = st.autorefresh(interval=200, limit=None, key="route_autorefresh")
    st.session_state.t = min(t_max, st.session_state.t + step_sec * speed_mult)

# Build map
deck_obj = build_deck(df, st.session_state.t)
if deck_obj is None:
    st.stop()

deck, cur = deck_obj

# Show map + current reading
left, right = st.columns([2, 1], gap="large")
with left:
    st.pydeck_chart(deck, use_container_width=True)

with right:
    st.subheader("Current Point")
    row = cur.iloc[0].to_dict()

    # Pretty time if available
    if row.get("time_dt_utc") is not None and pd.notna(row.get("time_dt_utc")):
        st.write(f"**UTC Time:** {row['time_dt_utc']}")
    st.write(f"**seconds_elapsed:** {row.get('seconds_elapsed')}")
    st.write(f"**speed:** {row.get('speed')}")
    st.write(f"**bearing:** {row.get('bearing')}")
    st.write(f"**latitude:** {row.get('latitude')}")
    st.write(f"**longitude:** {row.get('longitude')}")

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
