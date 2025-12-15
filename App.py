import streamlit as st
import pandas as pd
import numpy as np
import json
from io import StringIO
import streamlit.components.v1 as components

st.set_page_config(page_title="Multi-Group Map Playback", layout="wide")

# -----------------------------
# Sample generator (16 groups)
# -----------------------------
def generate_multi_group_sample(
    n_groups=16,
    n_points=120,
    start_lat=21.2710244,
    start_lon=78.5795227,
    dt=1.0,
    jitter_m=6.0,
    seed=42
) -> pd.DataFrame:
    """
    Generates a combined CSV-like dataset:
    group, seconds_elapsed, latitude, longitude, bearing
    using small random-walk GPS tracks around a start point.
    """
    rng = np.random.default_rng(seed)

    # meters -> degrees approx
    # 1 deg lat ~ 111_320 m
    m_to_deg_lat = 1 / 111_320
    # 1 deg lon ~ 111_320*cos(lat)
    m_to_deg_lon = 1 / (111_320 * np.cos(np.radians(start_lat)))

    rows = []
    for g in range(1, n_groups + 1):
        # Slightly different starting points per group
        lat = start_lat + rng.normal(0, 30 * m_to_deg_lat)
        lon = start_lon + rng.normal(0, 30 * m_to_deg_lon)

        seconds = 0.0
        prev_lat, prev_lon = lat, lon

        # give each group a drift direction
        drift_angle = rng.uniform(0, 2*np.pi)
        drift_lat = np.cos(drift_angle) * (2.0 * m_to_deg_lat)
        drift_lon = np.sin(drift_angle) * (2.0 * m_to_deg_lon)

        for i in range(n_points):
            # random movement (jitter) + drift
            step_lat = rng.normal(0, jitter_m * m_to_deg_lat) + drift_lat
            step_lon = rng.normal(0, jitter_m * m_to_deg_lon) + drift_lon

            lat = lat + step_lat
            lon = lon + step_lon

            # bearing from prev->curr
            dy = np.radians(lat - prev_lat)
            dx = np.radians(lon - prev_lon) * np.cos(np.radians((lat + prev_lat) / 2))
            angle = np.degrees(np.arctan2(dx, dy))
            bearing = (angle + 360) % 360

            rows.append([g, seconds, lat, lon, bearing])

            prev_lat, prev_lon = lat, lon
            seconds += dt

    return pd.DataFrame(rows, columns=["group", "seconds_elapsed", "latitude", "longitude", "bearing"])


# -----------------------------
# Load / prepare user data
# -----------------------------
def load_df(uploaded, sheet_url: str | None) -> pd.DataFrame:
    if uploaded is not None:
        name = uploaded.name.lower()
        if name.endswith(".tsv"):
            return pd.read_csv(uploaded, sep="\t")
        return pd.read_csv(uploaded)
    if sheet_url and sheet_url.strip():
        return pd.read_csv(sheet_url.strip())
    # default sample: 16 groups
    return generate_multi_group_sample()

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Required for multi-group playback
    for c in ["group", "seconds_elapsed", "latitude", "longitude"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # coerce types
    df["group"] = pd.to_numeric(df["group"], errors="coerce").astype("Int64")
    df["seconds_elapsed"] = pd.to_numeric(df["seconds_elapsed"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    if "bearing" in df.columns:
        df["bearing"] = pd.to_numeric(df["bearing"], errors="coerce")

    df = df.dropna(subset=["group", "seconds_elapsed", "latitude", "longitude"])
    df["group"] = df["group"].astype(int)

    # sort within each group
    df = df.sort_values(["group", "seconds_elapsed"]).reset_index(drop=True)

    # bearing fallback per group if missing/mostly empty
    if "bearing" not in df.columns or df["bearing"].isna().mean() > 0.8:
        df["bearing"] = 0.0

        for g, gdf in df.groupby("group", sort=True):
            lat = np.radians(gdf["latitude"].to_numpy())
            lon = np.radians(gdf["longitude"].to_numpy())
            dlon = np.diff(lon, prepend=lon[0])
            y = np.sin(dlon) * np.cos(lat)
            x = np.cos(lat) * np.sin(lat) - np.sin(lat) * np.cos(lat) * np.cos(dlon)  # safe-ish fallback
            brng = (np.degrees(np.arctan2(y, x)) + 360) % 360
            df.loc[gdf.index, "bearing"] = np.nan_to_num(brng, nan=0.0)

    return df


# -----------------------------
# UI
# -----------------------------
st.title("🧭 Multi-Group Live Map Playback (16 cursors)")

with st.sidebar:
    st.subheader("Data Source")
    uploaded = st.file_uploader("Upload combined CSV/TSV (must include group column)", type=["csv", "tsv"])
    sheet_url = st.text_input("Google Sheet CSV export link (combined data)")

    st.divider()
    st.subheader("Create sample for students")
    n_groups = st.number_input("Groups", 1, 50, 16, 1)
    n_points = st.number_input("Points per group", 20, 2000, 150, 10)
    dt = st.number_input("Seconds step (dt)", 0.1, 10.0, 1.0, 0.1)

# Sample download
sample_df = generate_multi_group_sample(n_groups=int(n_groups), n_points=int(n_points), dt=float(dt))
st.download_button(
    "⬇ Download sample CSV (16 groups)",
    data=sample_df.to_csv(index=False).encode("utf-8"),
    file_name="location_multigroup_sample.csv",
    mime="text/csv"
)

# Load + prepare real df
try:
    raw = load_df(uploaded, sheet_url)
    df = prepare_df(raw)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

groups = sorted(df["group"].unique().tolist())
if len(groups) < 1:
    st.warning("No groups found.")
    st.stop()

# Palette (distinct)
palette = [
    "#e11d48", "#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#0ea5e9", "#db2777", "#22c55e",
    "#f97316", "#4f46e5", "#14b8a6", "#a855f7", "#ef4444", "#3b82f6", "#84cc16", "#eab308",
    "#06b6d4", "#fb7185", "#10b981", "#facc15"
]

# Build payload for JS: per-group coords/times/bearing
tracks = []
global_tmin = float(df["seconds_elapsed"].min())
global_tmax = float(df["seconds_elapsed"].max())

# center = first point of first group
first_g = groups[0]
first_row = df[df["group"] == first_g].iloc[0]
center = [float(first_row["longitude"]), float(first_row["latitude"])]

for i, g in enumerate(groups):
    gdf = df[df["group"] == g].copy()
    coords = gdf[["longitude", "latitude"]].astype(float).values.tolist()
    times = gdf["seconds_elapsed"].astype(float).values.tolist()
    bears = gdf["bearing"].astype(float).fillna(0).values.tolist()
    tracks.append({
        "group": int(g),
        "color": palette[i % len(palette)],
        "coords": coords,
        "times": times,
        "bearings": bears
    })

payload = {
    "center": center,
    "tMin": global_tmin,
    "tMax": global_tmax,
    "tracks": tracks
}
payload_json = json.dumps(payload)

# -----------------------------
# MapLibre HTML (multi cursor)
# -----------------------------
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
      width: 420px;
    }
    .row { display:flex; align-items:center; gap:10px; margin-top:8px; }
    .btn { padding:6px 10px; border-radius:10px; border:1px solid #ccc; background:#fff; cursor:pointer; user-select:none; }
    input[type="range"] { width: 100%; }
    .small { font-size: 12px; color:#333; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    .toggle { display:flex; align-items:center; gap:8px; margin-top:8px; }
    .legend { margin-top:8px; max-height:140px; overflow:auto; border-top:1px dashed #ddd; padding-top:8px; }
    .legend-item { display:flex; gap:8px; align-items:center; margin:4px 0; }
    .dot { width:10px; height:10px; border-radius:999px; }
  </style>
</head>
<body>
  <div id="map"></div>

  <div class="panel">
    <div class="row" style="justify-content:space-between;">
      <div>
        <b>Multi-Group Playback</b>
        <div class="small">All groups move together • different colors</div>
      </div>
      <div class="btn" id="btnPlay">▶ Play</div>
      <div class="btn" id="btnPause">⏸ Pause</div>
    </div>

    <div class="row">
      <div class="small" style="width:72px;">Speed</div>
      <input id="speed" type="range" min="0.25" max="8" value="1" step="0.25" />
      <div class="small mono" id="speedVal">1.00x</div>
    </div>

    <div class="row">
      <div class="small" style="width:72px;">Scrub</div>
      <input id="scrub" type="range" min="0" max="1000" value="0" step="1" />
    </div>

    <div class="toggle">
      <input id="follow" type="checkbox" checked />
      <div class="small">Follow (camera tracks the average position)</div>
    </div>

    <div class="row" style="justify-content:space-between;">
      <div class="small mono" id="timeVal"></div>
      <div class="small mono" id="posVal"></div>
    </div>

    <div class="legend" id="legend"></div>
  </div>

<script>
  const DATA = __PAYLOAD_JSON__;
  const tMin = DATA.tMin, tMax = DATA.tMax;

  // OSM raster basemap
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

  // UI
  const btnPlay = document.getElementById("btnPlay");
  const btnPause = document.getElementById("btnPause");
  const speedEl = document.getElementById("speed");
  const speedVal = document.getElementById("speedVal");
  const scrub = document.getElementById("scrub");
  const timeVal = document.getElementById("timeVal");
  const posVal = document.getElementById("posVal");
  const followEl = document.getElementById("follow");
  const legendEl = document.getElementById("legend");

  function scrubToT(s){
    const u = s/1000.0;
    return tMin + (tMax-tMin)*u;
  }
  function tToScrub(t){
    const u = (t-tMin)/(tMax-tMin);
    return Math.max(0, Math.min(1000, Math.round(u*1000)));
  }

  function lerp(a,b,t){ return a+(b-a)*t; }

  function findSegmentIndex(times, t){
    if(t <= times[0]) return 0;
    if(t >= times[times.length-1]) return times.length-2;
    let lo=0, hi=times.length-1;
    while(hi-lo>1){
      const mid=(lo+hi)>>1;
      if(times[mid] <= t) lo=mid; else hi=mid;
    }
    return lo;
  }

  function interpState(track, t){
    const times = track.times, coords = track.coords, bears = track.bearings;
    if(times.length < 2) {
      const p = coords[0];
      return { lon:p[0], lat:p[1], bearing: (bears[0] ?? 0), idx:0 };
    }
    if(t <= times[0]){
      const p = coords[0];
      return { lon:p[0], lat:p[1], bearing:(bears[0] ?? 0), idx:0 };
    }
    if(t >= times[times.length-1]){
      const p = coords[coords.length-1];
      return { lon:p[0], lat:p[1], bearing:(bears[bears.length-1] ?? 0), idx:coords.length-1 };
    }

    const i = findSegmentIndex(times, t);
    const t0 = times[i], t1 = times[i+1];
    const p0 = coords[i], p1 = coords[i+1];
    const b0 = (bears[i] ?? 0), b1 = (bears[i+1] ?? b0);

    const u = (t1===t0) ? 0 : (t - t0)/(t1 - t0);
    const lon = lerp(p0[0], p1[0], u);
    const lat = lerp(p0[1], p1[1], u);

    let db = ((b1-b0+540)%360)-180;
    const bearing = (b0 + db*u + 360) % 360;

    return { lon, lat, bearing, idx:i };
  }

  // Create markers + route layers per group
  const markers = [];
  const traveledSources = []; // source ids to update
  const fullSources = [];

  function makeArrow(color){
    const el = document.createElement("div");
    el.style.width = "18px";
    el.style.height = "18px";
    el.style.background = color;
    el.style.border = "2px solid white";
    el.style.borderRadius = "6px";
    el.style.boxShadow = "0 6px 18px rgba(0,0,0,0.25)";
    el.style.transformOrigin = "center";
    el.style.clipPath = "polygon(50% 0%, 100% 60%, 50% 45%, 0% 60%)";
    return el;
  }

  // Fit bounds to all tracks
  function fitToAll(){
    let minLon=Infinity, minLat=Infinity, maxLon=-Infinity, maxLat=-Infinity;
    for(const tr of DATA.tracks){
      for(const p of tr.coords){
        minLon=Math.min(minLon, p[0]); minLat=Math.min(minLat, p[1]);
        maxLon=Math.max(maxLon, p[0]); maxLat=Math.max(maxLat, p[1]);
      }
    }
    map.fitBounds([[minLon,minLat],[maxLon,maxLat]], { padding: 60, duration: 0 });
  }

  map.on("load", () => {
    fitToAll();

    // legend
    legendEl.innerHTML = "";
    for(const tr of DATA.tracks){
      const div = document.createElement("div");
      div.className = "legend-item";
      div.innerHTML = `<span class="dot" style="background:${tr.color}"></span><span class="small mono">Group ${tr.group}</span>`;
      legendEl.appendChild(div);
    }

    for(const tr of DATA.tracks){
      const gid = tr.group;
      const fullId = `route_full_${gid}`;
      const travId = `route_trav_${gid}`;

      // full route source/layer
      map.addSource(fullId, {
        type:"geojson",
        data:{ type:"Feature", geometry:{ type:"LineString", coordinates: tr.coords } }
      });
      map.addLayer({
        id:`layer_full_${gid}`,
        type:"line",
        source:fullId,
        paint:{
          "line-width": 4,
          "line-opacity": 0.35,
          "line-color": tr.color
        }
      });

      // traveled route source/layer (updates with time)
      map.addSource(travId, {
        type:"geojson",
        data:{ type:"Feature", geometry:{ type:"LineString", coordinates: [tr.coords[0]] } }
      });
      map.addLayer({
        id:`layer_trav_${gid}`,
        type:"line",
        source:travId,
        paint:{
          "line-width": 5,
          "line-opacity": 0.95,
          "line-color": tr.color
        }
      });

      // marker
      const el = makeArrow(tr.color);
      const m = new maplibregl.Marker({ element: el }).setLngLat(tr.coords[0]).addTo(map);
      markers.push({ group: gid, marker: m, el, track: tr, travId });
      traveledSources.push(travId);
      fullSources.push(fullId);
    }
  });

  // Playback
  let playing=false;
  let speed=parseFloat(speedEl.value);
  let t=tMin;
  let lastTs=null;

  function updateUI(avg){
    timeVal.textContent = `t=${t.toFixed(2)}s`;
    posVal.textContent = `avg lat=${avg.lat.toFixed(6)} lon=${avg.lon.toFixed(6)}`;
    speedVal.textContent = `${speed.toFixed(2)}x`;
  }

  // Update traveled route less frequently for performance
  let frameCounter = 0;

  function tick(ts){
    if(!playing) return;
    if(lastTs===null) lastTs=ts;
    const dtMs = ts-lastTs;
    lastTs=ts;

    t += (dtMs/1000.0)*speed;
    if(t>=tMax){ t=tMax; playing=false; }

    let sumLon=0, sumLat=0, count=0;

    for(const m of markers){
      const state = interpState(m.track, t);
      m.marker.setLngLat([state.lon, state.lat]);
      m.el.style.transform = `rotate(${state.bearing}deg)`;

      sumLon += state.lon; sumLat += state.lat; count++;

      // update traveled route occasionally
      if(frameCounter % 3 === 0 && map.getSource(m.travId)){
        const upto = Math.max(1, Math.min(m.track.coords.length, (state.idx + 2)));
        const coordsSlice = m.track.coords.slice(0, upto);
        map.getSource(m.travId).setData({
          type:"Feature",
          geometry:{ type:"LineString", coordinates: coordsSlice }
        });
      }
    }

    const avg = { lon: sumLon/count, lat: sumLat/count };
    if(followEl.checked){
      map.easeTo({ center:[avg.lon, avg.lat], duration: 180, easing:(x)=>x });
    }

    scrub.value = tToScrub(t);
    updateUI(avg);

    frameCounter++;
    requestAnimationFrame(tick);
  }

  btnPlay.onclick=()=>{
    if(!playing){ playing=true; lastTs=null; requestAnimationFrame(tick); }
  };
  btnPause.onclick=()=>{ playing=false; };

  speedEl.oninput=()=>{
    speed=parseFloat(speedEl.value);
    speedVal.textContent = `${speed.toFixed(2)}x`;
  };

  scrub.oninput=()=>{
    playing=false;
    t = scrubToT(parseInt(scrub.value,10));

    let sumLon=0, sumLat=0, count=0;
    for(const m of markers){
      const state = interpState(m.track, t);
      m.marker.setLngLat([state.lon, state.lat]);
      m.el.style.transform = `rotate(${state.bearing}deg)`;

      sumLon += state.lon; sumLat += state.lat; count++;

      if(map.getSource(m.travId)){
        const upto = Math.max(1, Math.min(m.track.coords.length, (state.idx + 2)));
        const coordsSlice = m.track.coords.slice(0, upto);
        map.getSource(m.travId).setData({
          type:"Feature",
          geometry:{ type:"LineString", coordinates: coordsSlice }
        });
      }
    }
    const avg = { lon: sumLon/count, lat: sumLat/count };
    map.jumpTo({ center:[avg.lon, avg.lat] });
    updateUI(avg);
  };

  // init UI
  updateUI({ lon: DATA.center[0], lat: DATA.center[1] });
</script>
</body>
</html>
"""

html = html.replace("__PAYLOAD_JSON__", payload_json)
components.html(html, height=800)

st.subheader("Preview (combined data)")
st.write("Columns needed: **group, seconds_elapsed, latitude, longitude** (bearing optional).")
st.dataframe(df.head(50), use_container_width=True)
