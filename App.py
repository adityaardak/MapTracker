import streamlit as st
import pandas as pd
import numpy as np
import json
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

# -----------------------------
# Sample data (16 groups)
# -----------------------------
def generate_sample():
    rows = []
    base_lat, base_lon = 21.2710244, 78.5795227
    rng = np.random.default_rng(42)

    for g in range(1, 17):
        lat, lon = base_lat + rng.normal(0, 0.0003), base_lon + rng.normal(0, 0.0003)
        for i in range(80):
            lat += rng.normal(0, 0.00005)
            lon += rng.normal(0, 0.00005)
            rows.append([g, lat, lon])
    return pd.DataFrame(rows, columns=["group", "latitude", "longitude"])

df = generate_sample()

# -----------------------------
# Compute roaming circles
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def make_circle(lon, lat, radius_m, points=64):
    radius_m = max(radius_m, 10)  # 🔴 IMPORTANT FIX
    lat_r = np.radians(lat)
    dlat = radius_m / 111320
    dlon = radius_m / (111320 * np.cos(lat_r))
    coords = []
    for a in np.linspace(0, 2*np.pi, points):
        coords.append([lon + dlon*np.cos(a), lat + dlat*np.sin(a)])
    coords.append(coords[0])
    return coords

palette = [
    "#e11d48", "#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#0ea5e9",
    "#db2777", "#22c55e", "#f97316", "#4f46e5", "#14b8a6", "#a855f7",
    "#ef4444", "#3b82f6", "#84cc16", "#eab308"
]

features = []

for i, g in enumerate(sorted(df.group.unique())):
    gdf = df[df.group == g]
    lat_c = gdf.latitude.mean()
    lon_c = gdf.longitude.mean()
    dist = haversine(lat_c, lon_c, gdf.latitude, gdf.longitude)
    radius = np.percentile(dist, 95)

    features.append({
        "type": "Feature",
        "properties": {
            "group": g,
            "color": palette[i % len(palette)],
            "radius": float(radius)
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [make_circle(lon_c, lat_c, radius)]
        }
    })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

payload = json.dumps({
    "geojson": geojson,
    "center": [float(df.longitude.mean()), float(df.latitude.mean())]
})

# -----------------------------
# MapLibre HTML (FIXED)
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
</style>
</head>
<body>
<div id="map"></div>

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
    layers: [
      { id: 'osm', type: 'raster', source: 'osm' }
    ]
  },
  center: DATA.center,
  zoom: 15
});

map.on('load', () => {
  map.addSource('groups', {
    type: 'geojson',
    data: DATA.geojson
  });

  map.addLayer({
    id: 'group-fill',
    type: 'fill',
    source: 'groups',
    paint: {
      'fill-color': ['get', 'color'],
      'fill-opacity': 0.25
    }
  });

  map.addLayer({
    id: 'group-line',
    type: 'line',
    source: 'groups',
    paint: {
      'line-color': ['get', 'color'],
      'line-width': 3
    }
  });

  const bounds = new maplibregl.LngLatBounds();
  DATA.geojson.features.forEach(f => {
    f.geometry.coordinates[0].forEach(p => bounds.extend(p));
  });
  map.fitBounds(bounds, { padding: 80 });
});
</script>
</body>
</html>
"""

html = html.replace("__DATA__", payload)

components.html(html, height=800)
