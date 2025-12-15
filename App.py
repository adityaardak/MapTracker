import streamlit as st
import pandas as pd
import numpy as np
import json
from io import StringIO
import streamlit.components.v1 as components

st.set_page_config(page_title="MapTracker - Smooth Playback", layout="wide")

SAMPLE_TSV = """time\tseconds_elapsed\tbearingAccuracy\tspeedAccuracy\tverticalAccuracy\thorizontalAccuracy\tspeed\tbearing\taltitude\tlongitude\tlatitude
1.74296E+18\t0.255999756\t45\t1.5\t1.307455301\t20.59600067\t0\t0\t374.8000183\t78.5795227\t21.2710244
1.74296E+18\t3.175305176\t56.90444946\t0.587098598\t1.336648345\t12.78999996\t0.530843079\t102.8688431\t374.8000183\t78.5796365\t21.2710753
1.74296E+18\t3.242290771\t71.24659729\t0.923150122\t1.337318182\t10.19900036\t0.546774924\t102.1160355\t374.8000183\t78.5796392\t21.2710753
1.74296E+18\t4.241882813\t89.88391113\t0.781779766\t1.347314119\t8.765999794\t0.213110328\t88.08830261\t374.8000183\t78.5796205\t21.2710833
"""

# ----------------------------- Data load -----------------------------
def load_df(uploaded, sheet_url: str | None) -> pd.DataFrame:
    if uploaded is not None:
        name = uploaded.name.lower()
        if name.endswith(".tsv"):
            return pd.read_csv(uploaded, sep="\t")
        return pd.read_csv(uploaded)
    if sheet_url and sheet_url.strip():
        return pd.read_csv(sheet_url.strip())
    return pd.read_csv(StringIO(SAMPLE_TSV), sep="\t")

# ----------------------------- Feature engineering (no ML) -----------------------------
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

    # required columns
    for c in ["seconds_elapsed", "latitude", "longitude"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # numeric coercion
    for c in ["seconds_elapsed", "latitude", "longitude", "bearing", "speed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["seconds_elapsed", "latitude", "longitude"]).sort_values("seconds_elapsed").reset_index(drop=True)

    # bearing fallback if missing
    if "bearing" not in df.columns or df["bearing"].isna().mean() > 0.8:
        lat1 = np.radians(df["latitude"].shift(1))
        lat2 = np.radians(df["latitude"])
        dlon = np.radians(df["longitude"] - df["longitude"].shift(1))
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        brng = (np.degrees(np.arctan2(y, x)) + 360) % 360
        df["bearing"] = brng.fillna(0)

    # movement features
    df["dt"] = df["seconds_elapsed"].diff()
    df["lat_prev"] = df["latitude"].shift(1)
    df["lon_prev"] = df["longitude"].shift(1)

    df["dist_m"] = haversine_m(df["lat_prev"], df["lon_prev"], df["latitude"], df["longitude"]).fillna(0)
    df.loc[0, "dist_m"] = 0

    # speed_calc (m/s)
    df["speed_calc"] = (df["dist_m"] / df["dt"]).replace([np.inf, -np.inf], np.nan).fillna(0)

    # accel (m/s^2)
    df["accel_mps2"] = (df["speed_calc"].diff() / df["dt"]).replace([np.inf, -np.inf], np.nan).fillna(0)

    # stop detection
    df["is_stop"] = df["speed_calc"] < 0.5

    # simple anomaly rule (no ML): unusually large jump in one step
    mu = df["dist_m"].mean()
    sd = df["dist_m"].std(ddof=0)
    thr = mu + 3 * (sd if sd > 0 else 1.0)
    df["anomaly_rule"] = df["dist_m"] > thr

    return df

# ----------------------------- UI -----------------------------
st.title("🗺️ Smooth GPS Playback (Play/Pause) + Analytics (No ML)")

with st.sidebar:
    st.subheader("Data Source")
    uploaded = st.file_uploader("Upload CSV / TSV", type=["csv", "tsv"])
    sheet_url = st.text_input("Google Sheet CSV export link")

try:
    raw = load_df(uploaded, sheet_url)
    df = prepare_df(raw)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

if len(df) < 2:
    st.warning("Need at least 2 valid points to animate.")
    st.dataframe(df)
    st.stop()

# KPIs
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Points", f"{len(df)}")
k2.metric("Total distance", f"{df['dist_m'].sum()/1000:.3f} km")
k3.metric("Avg speed", f"{df['speed_calc'].mean():.2f} m/s")
k4.metric("Stop %", f"{df['is_stop'].mean()*100:.1f}%")
k5.metric("Rule anomalies", f"{int(df['anomaly_rule'].sum())}")

coords = df[["longitude", "latitude"]].astype(float).values.tolist()
times = df["seconds_elapsed"].astype(float).values.tolist()
bearings = df["bearing"].astype(float).values.tolist()

payload = {
    "coords": coords,
    "times": times,
    "bearings": bearings,
    "center": [float(coords[0][0]), float(coords[0][1])],
    "tMin": float(min(times)),
    "tMax": float(max(times)),
}
payload_json = json.dumps(payload)

# Keep the same MapLibre + OSM raster basemap + smooth playback controls
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
    .row { display:flex; align-items:center; gap:10px; margin-top:8px; }
    .btn {
      padding:6px 10px; border-radius:10px; border:1px solid #ccc; background:#fff;
      cursor:pointer; user-select:none;
    }
    .btn:active { transform: translateY(1px); }
    input[type="range"] { width: 100%; }
    .small { font-size: 12px; color:#333; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    .toggle { display:flex; align-items:center; gap:8px; margin-top:8px; }
  </style>
</head>
<body>
  <div id="map"></div>

  <div class="panel">
    <div class="row" style="justify-content:space-between;">
      <div>
        <b>Smooth Playback</b>
        <div class="small">OSM roads/labels • browser-side animation</div>
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
      <div class="small">Follow marker (camera)</div>
    </div>

    <div class="row" style="justify-content:space-between;">
      <div class="small mono" id="timeVal"></div>
      <div class="small mono" id="posVal"></div>
    </div>
  </div>

<script>
  const DATA = __PAYLOAD_JSON__;
  const coords = DATA.coords;
  const times  = DATA.times;
  const bears  = DATA.bearings;

  function lerp(a,b,t){ return a+(b-a)*t; }

  function findSegmentIndex(t){
    if(t <= times[0]) return 0;
    if(t >= times[times.length-1]) return times.length-2;
    let lo=0, hi=times.length-1;
    while(hi-lo>1){
      const mid=(lo+hi)>>1;
      if(times[mid] <= t) lo=mid; else hi=mid;
    }
    return lo;
  }

  function interpState(t){
    const i = findSegmentIndex(t);
    const t0=times[i], t1=times[i+1];
    const p0=coords[i], p1=coords[i+1];
    const b0=(bears[i] ?? 0), b1=(bears[i+1] ?? b0);
    const u = (t1===t0)?0:(t-t0)/(t1-t0);
    const lon=lerp(p0[0], p1[0], u);
    const lat=lerp(p0[1], p1[1], u);
    let db = ((b1-b0+540)%360)-180;
    const bearing=(b0 + db*u + 360) % 360;
    return {lon, lat, bearing};
  }

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
    zoom: 16
  });
  map.addControl(new maplibregl.NavigationControl(), "top-right");

  // Fit to route bounds
  function fitToRoute(){
    let minLon=Infinity, minLat=Infinity, maxLon=-Infinity, maxLat=-Infinity;
    for(const p of coords){
      minLon=Math.min(minLon, p[0]); minLat=Math.min(minLat, p[1]);
      maxLon=Math.max(maxLon, p[0]); maxLat=Math.max(maxLat, p[1]);
    }
    map.fitBounds([[minLon, minLat],[maxLon, maxLat]], { padding: 60, duration: 0 });
  }

  // Marker arrow
  const el = document.createElement("div");
  el.style.width = "18px";
  el.style.height = "18px";
  el.style.background = "#e11d48";
  el.style.border = "2px solid white";
  el.style.borderRadius = "6px";
  el.style.boxShadow = "0 6px 18px rgba(0,0,0,0.25)";
  el.style.transformOrigin = "center";
  el.style.clipPath = "polygon(50% 0%, 100% 60%, 50% 45%, 0% 60%)";

  const marker = new maplibregl.Marker({ element: el }).setLngLat(coords[0]).addTo(map);

  // Route line overlay
  const lineGeo = { type:"Feature", geometry:{ type:"LineString", coordinates: coords } };

  map.on("load", () => {
    fitToRoute();
    map.addSource("route", { type:"geojson", data: lineGeo });
    map.addLayer({
      id:"route-line",
      type:"line",
      source:"route",
      paint:{ "line-width": 5, "line-opacity": 0.9 }
    });
  });

  // UI
  const btnPlay = document.getElementById("btnPlay");
  const btnPause = document.getElementById("btnPause");
  const speedEl = document.getElementById("speed");
  const speedVal = document.getElementById("speedVal");
  const scrub = document.getElementById("scrub");
  const timeVal = document.getElementById("timeVal");
  const posVal = document.getElementById("posVal");
  const followEl = document.getElementById("follow");

  const tMin = DATA.tMin, tMax = DATA.tMax;

  function scrubToT(s){
    const u = s/1000.0;
    return tMin + (tMax-tMin)*u;
  }
  function tToScrub(t){
    const u = (t-tMin)/(tMax-tMin);
    return Math.max(0, Math.min(1000, Math.round(u*1000)));
  }

  let playing=false;
  let speed=parseFloat(speedEl.value);
  let t=tMin;
  let lastTs=null;

  function updateUI(state){
    timeVal.textContent = `t=${t.toFixed(2)}s`;
    posVal.textContent = `lat=${state.lat.toFixed(6)} lon=${state.lon.toFixed(6)}`;
    speedVal.textContent = `${speed.toFixed(2)}x`;
  }

  function tick(ts){
    if(!playing) return;
    if(lastTs===null) lastTs=ts;
    const dtMs = ts-lastTs;
    lastTs=ts;

    t += (dtMs/1000.0)*speed;
    if(t>=tMax){ t=tMax; playing=false; }

    const state = interpState(t);
    marker.setLngLat([state.lon, state.lat]);
    el.style.transform = `rotate(${state.bearing}deg)`;

    if(followEl.checked){
      map.easeTo({ center:[state.lon, state.lat], duration: 180, easing:(x)=>x });
    }

    scrub.value = tToScrub(t);
    updateUI(state);
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
    const state = interpState(t);
    marker.setLngLat([state.lon, state.lat]);
    el.style.transform = `rotate(${state.bearing}deg)`;
    map.jumpTo({ center:[state.lon, state.lat] });
    updateUI(state);
  };

  updateUI(interpState(t));
</script>
</body>
</html>
"""

html = html.replace("__PAYLOAD_JSON__", payload_json)

left, right = st.columns([2, 1], gap="large")

with left:
    components.html(html, height=780)

with right:
    st.subheader("📊 Analytics (No ML)")

    st.write("Speed (calculated) vs Time")
    st.line_chart(df.set_index("seconds_elapsed")["speed_calc"], height=220)

    st.write("Acceleration vs Time")
    st.line_chart(df.set_index("seconds_elapsed")["accel_mps2"], height=220)

    st.write("Step Distance (m)")
    st.bar_chart(df["dist_m"], height=220)

    st.write("Rule-based anomaly points (big jumps)")
    st.dataframe(
        df[df["anomaly_rule"]][["seconds_elapsed", "dist_m", "speed_calc", "latitude", "longitude"]].head(50),
        use_container_width=True
    )

    st.download_button(
        "⬇ Download enriched CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="gps_enriched_no_ml.csv",
        mime="text/csv"
    )

with st.expander("Preview Data"):
    st.dataframe(df.head(80), use_container_width=True)
