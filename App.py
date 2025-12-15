import streamlit as st
import pandas as pd
import numpy as np
import json
import streamlit.components.v1 as components

st.set_page_config(layout="wide")
st.title("🟠 Group Roaming Radius Circles (Different Colors)")

# -----------------------------
# Sample data (16 groups)
# -----------------------------
def generate_sample(n_groups=16, n_points=120, seed=42):
    rows = []
    base_lat, base_lon = 21.2710244, 78.5795227
    rng = np.random.default_rng(seed)

    for g in range(1, n_groups + 1):
        lat = base_lat + float(rng.normal(0, 0.0003))
        lon = base_lon + float(rng.normal(0, 0.0003))
        for _ in range(n_points):
            lat += float(rng.normal(0, 0.00005))
            lon += float(rng.normal(0, 0.00005))
            rows.append([int(g), float(lat), float(lon)])
    return pd.DataFrame(rows, columns=["group", "latitude", "longitude"])

df = generate_sample()

# -----------------------------
# Helpers
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def make_circle(lon, lat, radius_m, points=72):
    """Return polygon ring coords as Python lists."""
    radius_m = float(max(radius_m, 10.0))
    lat_r = np.radians(lat)
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * np.cos(lat_r) + 1e-12)

    ring = []
    for a in np.linspace(0, 2*np.pi, points, endpoint=False):
        ring.append([float(lon + dlon*np.cos(a)), float(lat + dlat*np.sin(a))])
    ring.append(ring[0])  # close ring
    return ring

palette = [
    "#e11d48", "#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#0ea5e9",
    "#db2777", "#22c55e", "#f97316", "#4f46e5", "#14b8a6", "#a855f7",
    "#ef4444", "#3b82f6", "#84cc16", "#eab308"
]

# -----------------------------
# Build GeoJSON circles (ALL pure Python types)
# -----------------------------
features = []
for i, g in enumerate(sorted(df["group"].unique())):
    gdf = df[df["group"] == g]
    lat_c = float(gdf["latitude"].mean())
    lon_c = float(gdf["longitude"].mean())

    dist = haversine(lat_c, lon_c, gdf["latitude"].to_numpy(), gdf["longitude"].to_numpy())
    radius = float(np.percentile(dist, 95))  # meters

    features.append({
        "type": "Feature",
        "properties": {
            "group": int(g),
            "color": str(palette[i % len(palette)]),
            "radius_m": float(radius)
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [make_circle(lon_c, lat_c, radius)]
        }
    })

geojson = {"type": "FeatureCollection", "features": features}

center = [float(df["longitude"].mean()), float(df["latitude"].mean())]

# ✅ THIS NOW WORKS because everything is pure Python
payload = json.dumps({"geojson": geojson, "center": center})

# -----------------------------
# MapLibre HTML
# -----------------------------
html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet"/>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<style>
html, body, #map { margin:0; padding:0; height:100%; width:100%; }
.panel {
  position:absolute; top:12px; left:12px; z-index:10;
  background: rgba(255,255,255,0.95); border:1px solid #ddd;
  border-radius:12px; padding:10px 12px;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
  box-shadow: 0 6px 22px rgba(0,0,0,0.12);
}
.small { font-size: 12px; color:#333; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <b>Roaming Circles</b>
  <div class="small">Click a circle to see group + radius.</div>
</div>

<script>
const DATA = __DATA__;

const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: {
      osm: {
        type: 'raster',
        tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
        tileSize: 256
      }
    },
    layers: [{ id:'osm', type:'raster', source:'osm' }]
  },
  center: DATA.center,
  zoom: 15
});

map.addControl(new maplibregl.NavigationControl(), "top-right");

map.on('load', () => {
  map.addSource('groups', { type:'geojson', data: DATA.geojson });

  map.addLayer({
    id:'group-fill',
    type:'fill',
    source:'groups',
    paint:{
      'fill-color': ['get','color'],
      'fill-opacity': 0.25
    }
  });

  map.addLayer({
    id:'group-line',
    type:'line',
    source:'groups',
    paint:{
      'line-color': ['get','color'],
      'line-width': 3,
      'line-opacity': 0.95
    }
  });

  // Fit bounds to all circles
  const bounds = new maplibregl.LngLatBounds();
  DATA.geojson.features.forEach(f => {
    f.geometry.coordinates[0].forEach(p => bounds.extend(p));
  });
  map.fitBounds(bounds, { padding: 80, duration: 0 });

  // Popup on click
  map.on('click', 'group-fill', (e) => {
    const p = e.features[0].properties;
    const msg = `Group ${p.group} • radius ≈ ${Number(p.radius_m).toFixed(1)} m`;
    new maplibregl.Popup().setLngLat(e.lngLat).setHTML(`<b>${msg}</b>`).addTo(map);
  });

  map.on('mouseenter', 'group-fill', () => map.getCanvas().style.cursor = 'pointer');
  map.on('mouseleave', 'group-fill', () => map.getCanvas().style.cursor = '');
});
</script>
</body>
</html>
"""

html = html.replace("__DATA__", payload)
components.html(html, height=820)

with st.expander("Centers + radius (debug table)"):
    centers = []
    for f in geojson["features"]:
        centers.append({
            "group": f["properties"]["group"],
            "color": f["properties"]["color"],
            "radius_m": f["properties"]["radius_m"]
        })
    st.dataframe(pd.DataFrame(centers), use_container_width=True)
