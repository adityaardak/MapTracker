import streamlit as st
import pandas as pd
import numpy as np
import json
from io import StringIO
import streamlit.components.v1 as components

st.set_page_config(page_title="Group Roaming Radius Map", layout="wide")

# -----------------------------
# Sample multigroup data generator (same location base)
# -----------------------------
def generate_multi_group_sample(
    n_groups=16,
    n_points=120,
    start_lat=21.2710244,
    start_lon=78.5795227,
    dt=1.0,
    jitter_m=8.0,
    seed=42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m_to_deg_lat = 1 / 111_320
    m_to_deg_lon = 1 / (111_320 * np.cos(np.radians(start_lat)))

    rows = []
    for g in range(1, n_groups + 1):
        lat = start_lat + rng.normal(0, 25 * m_to_deg_lat)
        lon = start_lon + rng.normal(0, 25 * m_to_deg_lon)

        seconds = 0.0
        prev_lat, prev_lon = lat, lon

        drift_angle = rng.uniform(0, 2*np.pi)
        drift_lat = np.cos(drift_angle) * (1.8 * m_to_deg_lat)
        drift_lon = np.sin(drift_angle) * (1.8 * m_to_deg_lon)

        for _ in range(n_points):
            lat += rng.normal(0, jitter_m * m_to_deg_lat) + drift_lat
            lon += rng.normal(0, jitter_m * m_to_deg_lon) + drift_lon

            # bearing from prev -> curr (rough)
            dy = np.radians(lat - prev_lat)
            dx = np.radians(lon - prev_lon) * np.cos(np.radians((lat + prev_lat) / 2))
            angle = np.degrees(np.arctan2(dx, dy))
            bearing = (angle + 360) % 360

            rows.append([g, seconds, lat, lon, bearing])
            prev_lat, prev_lon = lat, lon
            seconds += dt

    return pd.DataFrame(rows, columns=["group", "seconds_elapsed", "latitude", "longitude", "bearing"])

# -----------------------------
# Load / prepare
# -----------------------------
SAMPLE_TSV = """group\tseconds_elapsed\tlongitude\tlatitude\tbearing
1\t0\t78.5795227\t21.2710244\t0
1\t1\t78.5796365\t21.2710753\t102
1\t2\t78.5796392\t21.2710753\t102
1\t3\t78.5796205\t21.2710833\t88
"""

def load_df(uploaded, sheet_url: str | None) -> pd.DataFrame:
    if uploaded is not None:
        name = uploaded.name.lower()
        if name.endswith(".tsv"):
            return pd.read_csv(uploaded, sep="\t")
        return pd.read_csv(uploaded)
    if sheet_url and sheet_url.strip():
        return pd.read_csv(sheet_url.strip())
    # default: generate 16 groups
    return generate_multi_group_sample()

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for c in ["group", "latitude", "longitude"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["group"] = pd.to_numeric(df["group"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    if "seconds_elapsed" in df.columns:
        df["seconds_elapsed"] = pd.to_numeric(df["seconds_elapsed"], errors="coerce")

    df = df.dropna(subset=["group", "latitude", "longitude"]).copy()
    df["group"] = df["group"].astype(int)

    # sort if time exists
    if "seconds_elapsed" in df.columns:
        df = df.sort_values(["group", "seconds_elapsed"]).reset_index(drop=True)
    else:
        df = df.sort_values(["group"]).reset_index(drop=True)

    return df

# -----------------------------
# Geo helpers
# -----------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def circle_polygon(lon, lat, radius_m, n=64):
    """
    Returns polygon coords (lon,lat) approximating a circle on Earth.
    Uses simple equirectangular approximation, good for small radii.
    """
    lat_r = np.radians(lat)
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * np.cos(lat_r) + 1e-12)

    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    coords = []
    for a in angles:
        coords.append([lon + dlon * np.cos(a), lat + dlat * np.sin(a)])
    coords.append(coords[0])  # close ring
    return coords

# -----------------------------
# UI
# -----------------------------
st.title("🟠 Group Roaming Radius Circles (Same Map, Different Colors)")

with st.sidebar:
    st.subheader("Data Source")
    uploaded = st.file_uploader("Upload CSV/TSV (must include group, latitude, longitude)", type=["csv", "tsv"])
    sheet_url = st.text_input("Google Sheet CSV export link")

    st.divider()
    st.subheader("Circle radius rule")
    mode = st.selectbox("Radius based on", ["95th percentile (recommended)", "Max distance"])
    fill_opacity = st.slider("Circle fill opacity", 0.0, 0.8, 0.18, 0.02)
    outline_opacity = st.slider("Outline opacity", 0.1, 1.0, 0.85, 0.05)

    st.divider()
    st.subheader("Generate sample for 16 groups")
    n_groups = st.number_input("Groups", 1, 50, 16, 1)
    n_points = st.number_input("Points per group", 30, 2000, 150, 10)

sample_df = generate_multi_group_sample(n_groups=int(n_groups), n_points=int(n_points))
st.download_button(
    "⬇ Download sample CSV with group column",
    data=sample_df.to_csv(index=False).encode("utf-8"),
    file_name="multigroup_roaming_sample.csv",
    mime="text/csv"
)

# Load + compute circles
try:
    df = prepare_df(load_df(uploaded, sheet_url))
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

groups = sorted(df["group"].unique())
if len(groups) == 0:
    st.warning("No groups found.")
    st.stop()

palette = [
    "#e11d48", "#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#0ea5e9", "#db2777", "#22c55e",
    "#f97316", "#4f46e5", "#14b8a6", "#a855f7", "#ef4444", "#3b82f6", "#84cc16", "#eab308",
    "#06b6d4", "#fb7185", "#10b981", "#facc15"
]

features = []
centers_table = []

for i, g in enumerate(groups):
    gdf = df[df["group"] == g]
    lat_c = float(gdf["latitude"].mean())
    lon_c = float(gdf["longitude"].mean())
    dists = haversine_m(lat_c, lon_c, gdf["latitude"].to_numpy(), gdf["longitude"].to_numpy())

    if mode.startswith("95"):
        r = float(np.percentile(dists, 95))
    else:
        r = float(np.max(dists)) if len(dists) else 0.0

    color = palette[i % len(palette)]
    poly = circle_polygon(lon_c, lat_c, max(r, 1.0), n=72)

    features.append({
        "type": "Feature",
        "properties": {"group": int(g), "color": color, "radius_m": r},
        "geometry": {"type": "Polygon", "coordinates": [poly]}
    })

    centers_table.append({"group": int(g), "center_lat": lat_c, "center_lon": lon_c, "radius_m": r})

geojson = {"type": "FeatureCollection", "features": features}
centers_df = pd.DataFrame(centers_table).sort_values("group")

# Map center
center = [centers_df["center_lon"].iloc[0], centers_df["center_lat"].iloc[0]]
payload_json = json.dumps({
    "center": center,
    "circles": geojson,
    "fillOpacity": float(fill_opacity),
    "outlineOpacity": float(outline_opacity),
})

# Map HTML (MapLibre + OSM raster + per-feature color)
html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
  <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
  <style>
    html, body { margin:0; padding:0; height:100%; width:100%; background:#fff; }
    #map { position:absolute; top:0; bottom:0; width:100%; }
    .panel {
      position:absolute; top:12px; left:12px; z-index:10;
      background: rgba(255,255,255,0.96); border:1px solid #ddd;
      border-radius:12px; padding:10px 12px;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
      box-shadow: 0 6px 22px rgba(0,0,0,0.12);
      width: 360px;
    }
    .small { font-size: 12px; color:#333; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel">
    <b>Roaming Radius Circles</b>
    <div class="small">Each circle shows the area a group is roaming in (radius computed from points).</div>
    <div class="small mono" style="margin-top:6px;">Tip: click a circle to see group + radius.</div>
  </div>

<script>
  const DATA = __PAYLOAD_JSON__;

  const style = {
    "version": 8,
    "sources": {
      "osm": {
        "type": "raster",
        "tiles": ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        "tileSize": 256,
        "attribution": "© OpenStreetMap contributors"
      }
    },
    "layers": [{ "id": "osm", "type": "raster", "source": "osm" }]
  };

  const map = new maplibregl.Map({
    container: "map",
    style,
    center: DATA.center,
    zoom: 15
  });
  map.addControl(new maplibregl.NavigationControl(), "top-right");

  map.on("load", () => {
    map.addSource("circles", { type: "geojson", data: DATA.circles });

    // Filled polygons
    map.addLayer({
      id: "circle-fill",
      type: "fill",
      source: "circles",
      paint: {
        "fill-color": ["get", "color"],
        "fill-opacity": DATA.fillOpacity
      }
    });

    // Outline
    map.addLayer({
      id: "circle-line",
      type: "line",
      source: "circles",
      paint: {
        "line-color": ["get", "color"],
        "line-width": 3,
        "line-opacity": DATA.outlineOpacity
      }
    });

    // Fit bounds to all circles
    const coordsAll = [];
    for (const f of DATA.circles.features){
      const ring = f.geometry.coordinates[0];
      for (const p of ring) coordsAll.push(p);
    }
    let minLon=Infinity, minLat=Infinity, maxLon=-Infinity, maxLat=-Infinity;
    for (const p of coordsAll){
      minLon=Math.min(minLon, p[0]); minLat=Math.min(minLat, p[1]);
      maxLon=Math.max(maxLon, p[0]); maxLat=Math.max(maxLat, p[1]);
    }
    map.fitBounds([[minLon,minLat],[maxLon,maxLat]], { padding: 70, duration: 0 });

    // Popup on click
    map.on("click", "circle-fill", (e) => {
      const p = e.features[0].properties;
      const msg = `Group ${p.group} • radius ≈ ${Number(p.radius_m).toFixed(1)} m`;
      new maplibregl.Popup()
        .setLngLat(e.lngLat)
        .setHTML(`<b>${msg}</b>`)
        .addTo(map);
    });

    map.on("mouseenter", "circle-fill", () => map.getCanvas().style.cursor = "pointer");
    map.on("mouseleave", "circle-fill", () => map.getCanvas().style.cursor = "");
  });
</script>
</body>
</html>
"""

html = html.replace("__PAYLOAD_JSON__", payload_json)

components.html(html, height=820)

st.subheader("Group Centers + Radius (meters)")
st.dataframe(centers_df, use_container_width=True)

st.download_button(
    "⬇ Download centers + radius table",
    data=centers_df.to_csv(index=False).encode("utf-8"),
    file_name="group_roaming_radius.csv",
    mime="text/csv"
)
