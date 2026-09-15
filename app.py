import base64
import io
import json
from pathlib import Path
from typing import Tuple

import folium
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from streamlit_folium import st_folium
from branca.element import MacroElement
from jinja2 import Template
import branca.colormap as cm
import matplotlib.colors as mcolors
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol
from scipy.ndimage import gaussian_filter
#from rasterio.transform import rowcol

VARIABLE_INFO = {

    "chla": {
        "name": "Chlorophyll-a concentration",
        "unit": "mg m⁻³",
        "description": "Indicator of phytoplankton biomass.",
    },

    "chlorophyll": {
        "name": "Chlorophyll-a concentration",
        "unit": "mg m⁻³",
        "description": "Indicator of phytoplankton biomass.",
    },

    "oxy": {
        "name": "Dissolved oxygen concentration",
        "unit": "mmol m⁻³",
        "description": "Dissolved oxygen concentration in seawater.",
    },

    "oxygen": {
        "name": "Dissolved oxygen concentration",
        "unit": "mmol m⁻³",
        "description": "Dissolved oxygen concentration in seawater.",
    },

    "salt": {
        "name": "Salinity",
        "unit": "PSU",
        "description": "Practical salinity of seawater.",
    },

    "salinity": {
        "name": "Salinity",
        "unit": "PSU",
        "description": "Practical salinity of seawater.",
    },

    "temp": {
        "name": "Temperature",
        "unit": "°C",
        "description": "Sea water temperature.",
    },

    "temperature": {
        "name": "Temperature",
        "unit": "°C",
        "description": "Sea water temperature.",
    },

    "mussel_weight": {
        "name": "Blue mussel fresh weight",
        "unit": "g individual⁻¹",
        "description": "Predicted biomass from the DEB model.",
    },
}

def _rgba_to_data_url(rgba: np.ndarray) -> str:
    im = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

def _clean(arr, src, variable=None):

    arr = arr.astype(np.float32)

    if src.nodata is not None:
        arr = np.where(arr == src.nodata, np.nan, arr)

    arr = np.where(np.isfinite(arr), arr, np.nan)

    # Remove undeclared fill values
    arr = np.where(np.abs(arr) > 1e30, np.nan, arr)

    return arr
def _render_preview_from_tif(
    tif_bytes: bytes,
    turbine_df: pd.DataFrame,
    *,
    max_size: int = 1024,
) -> Tuple[
    np.ndarray,
    Tuple[float, float, float, float],
    dict,
    float,
    float,
]:

    with MemoryFile(tif_bytes) as mem:
        with mem.open() as src:
            if src.crs is None or str(src.crs).upper() != "EPSG:4326":
                raise ValueError(f"Expected EPSG:4326 GeoTIFF; got {src.crs}.")

            scale = min(max_size / src.width, max_size / src.height, 1.0)
            out_w = max(1, int(src.width * scale))
            out_h = max(1, int(src.height * scale))
            data = src.read(
                out_shape=(src.count, out_h, out_w),
                resampling=Resampling.nearest,
                masked=True,
            )
            
            west = min(src.bounds.left, src.bounds.right)
            east = max(src.bounds.left, src.bounds.right)
            south = min(src.bounds.bottom, src.bounds.top)
            north = max(src.bounds.bottom, src.bounds.top)
            transform = src.transform

            if src.transform.e > 0:
                data = data[..., ::-1, :]
            

    if data.shape[0] >= 3:
        rgb = np.stack([data[0], data[1], data[2]], axis=-1).astype(np.float32)
        if np.ma.isMaskedArray(rgb):
            alpha = (~np.any(rgb.mask, axis=-1)).astype(np.uint8) * 255
            rgb = rgb.filled(0)
        else:
            alpha = np.full((rgb.shape[0], rgb.shape[1]), 255, dtype=np.uint8)
        if rgb.max() > 0:
            rgb = rgb / rgb.max()
        rgba = np.dstack([(rgb * 255).clip(0, 255).astype(np.uint8), alpha])
        style = {"kind": "rgb"}
    else:
        band = data[0].astype(np.float32)

# --- handle masked + nodata ---
        if np.ma.isMaskedArray(band):
            band = band.filled(np.nan)

        if src.nodata is not None:
            band = np.where(band == src.nodata, np.nan, band)

# --- clean invalid values ---
        band = np.where(np.isfinite(band), band, np.nan)

# --- optional: clip extreme garbage ---
        band = np.clip(band, 0, 1e6)

# --- compute mask for transparency ---
        mask = ~np.isfinite(band)

# --- robust percentiles ---
#        vmin = np.nanpercentile(band, 5)
#        vmax = np.nanpercentile(band, 95)

        # --------------------------------------------------
# Compute colour scale only around the OWF
# --------------------------------------------------



        buffer_deg = 0.05

        min_lon = turbine_df["lon"].min() - buffer_deg
        max_lon = turbine_df["lon"].max() + buffer_deg

        min_lat = turbine_df["lat"].min() - buffer_deg
        max_lat = turbine_df["lat"].max() + buffer_deg

# Convert lon/lat -> raster rows/cols
        r0, c0 = rowcol(transform, min_lon, max_lat)
        r1, c1 = rowcol(transform, max_lon, min_lat)

        r0, r1 = sorted((r0, r1))
        c0, c1 = sorted((c0, c1))

# Keep indices inside raster
        r0 = max(r0, 0)
        c0 = max(c0, 0)

        r1 = min(r1, band.shape[0])
        c1 = min(c1, band.shape[1])

        window = band[r0:r1, c0:c1]

        vmin = np.nanpercentile(window, 5)
        vmax = np.nanpercentile(window, 95)

# Fall back to whole raster if window is empty
        if (
            not np.isfinite(vmin)
            or not np.isfinite(vmax)
            or vmin == vmax
        ):
            vmin = np.nanpercentile(band, 5)
            vmax = np.nanpercentile(band, 95)


# --- render ALWAYS (not inside fallback!) ---
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        rgba_f = matplotlib.colormaps.get_cmap("viridis")(norm(band))
        rgba = (rgba_f * 255).astype(np.uint8)

# apply transparency
        rgba[mask, 3] = 0

        style = {"kind": "singleband"}


    return rgba, (south, west, north, east), style, vmin, vmax
    #vmin, vmax = 0, 255

def _load_geojson_timeseries(geojson_path: Path) -> pd.DataFrame:
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    rows = []
    for f in features:
        props = f.get("properties", {})
        geom = f.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        val = props.get("bwmus")
        t = props.get("time")
        if val is None or t is None:
            continue
        rows.append(
            {
                "time": str(t),
                #"chl_a": float(val),
                "value": float(val),
                "lon": float(coords[0]) if coords and coords[0] is not None else np.nan,
                "lat": float(coords[1]) if len(coords) > 1 and coords[1] is not None else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["time", "value", "lon", "lat"])

    df = pd.DataFrame(rows)
    time_num = pd.to_numeric(df["time"], errors="coerce")
    if time_num.notna().any():
        df = df.assign(_time_num=time_num).sort_values("_time_num").drop(columns=["_time_num"])
    else:
        df = df.sort_values("time")
    return df
    
@st.cache_data(show_spinner=False)
def _cached_load_geojson_timeseries(path_str: str) -> pd.DataFrame:
    return _load_geojson_timeseries(Path(path_str))


def _select_nearest_point(points_df: pd.DataFrame, click_lon: float, click_lat: float) -> pd.Series:
    d2 = (points_df["lon"] - float(click_lon)) ** 2 + (points_df["lat"] - float(click_lat)) ** 2
    return points_df.loc[d2.idxmin()]

if "selected_scenarios" not in st.session_state:
    st.session_state.selected_scenarios = []

st.set_page_config(
    page_title="OWF & LTA co-use",
    layout="wide"
)

#st.title("OWF & LTA co-use dashboard")
st.title("Offshore Co-use Decision Dashboard")
st.caption("Mussel biomass, scenarios, and spatial analysis viewer")


with st.expander("ℹ About this application", expanded=False):

    st.markdown("""
### Why?

This digital twin demonstrates how offshore wind farms and low-trophic
aquaculture can be planned jointly using scenario-based environmental
simulations.

The application supports transparent comparison of alternative offshore
co-use scenarios and illustrates the role of environmental digital twins
for adaptive marine spatial planning.

---

### What?

The dashboard visualizes outputs from a coupled

- Hydrodynamic model
- Wave model
- Biogeochemical model
- Dynamic Energy Budget (DEB) mussel growth model

for the Meerwind Offshore Wind Farm (German Bight).

Users can compare scenarios, variables and simulation dates to evaluate
aquaculture performance and environmental response.

---

### How?

The workflow consists of

Hydrodynamic forcing
→ Ecosystem simulation
→ Mussel growth modelling
→ Raster generation
→ Interactive visualization in Streamlit.
""")

base_dir = Path(__file__).parent

geojson_files = {
    "Scenario 1": base_dir / "harvest_timeseries_scenario_Scen_M2.geojson",
    "Scenario 2": base_dir / "harvest_timeseries_scenario_Scen_M3.geojson",
    "Scenario N": base_dir / "harvest_timeseries_scenario_Scen_N.geojson",
    "Scenario E": base_dir / "harvest_timeseries_scenario_Scen_E.geojson",
    "Scenario S": base_dir / "harvest_timeseries_scenario_Scen_S.geojson",
    "Scenario W": base_dir / "harvest_timeseries_scenario_Scen_W.geojson",
   # "Scenario 3": base_dir / "scenario3.geojson",
}

# --- Scan all tif files and organize ---
#tif_files = sorted((base_dir / "geotiff").glob("**/*.tif"))
# --- Scan all tif files and organize ---

tif_dirs = {
    "scenario": base_dir / "geotiff",
    "baseline_salt": base_dir / "salt_geotiff_ScenM0",
    "baseline_temp": base_dir / "temp_geotiff_ScenM0",
}

tif_files = []

for folder in tif_dirs.values():
    if folder.exists():
        tif_files.extend(sorted(folder.glob("**/*.tif")))


SCENARIO_TO_FOLDER = {
    "Scenario 1": "ScenM2",
    "Scenario 2": "ScenM3",
    "Scenario N": "Scen_N",
    "Scenario E": "Scen_E",
    "Scenario S": "Scen_S",
    "Scenario W": "Scen_W",
}

# --- wind turbine locations ---


turbine_csv = base_dir / "Meerwind_monopiles_lonlat.csv"

@st.cache_data(show_spinner=False)
def load_turbines(csv_path):
    df = pd.read_csv(csv_path, header=None, sep=r"\s+", engine="python")
    df = df.iloc[:, :2]
    df.columns = ["lon", "lat"]

    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    return df.dropna()

turbine_df = load_turbines(turbine_csv)

records = []

for f in tif_files:

    name = f.stem
    folder = f.parent.name          # <-- NEW

    try:
        var, tstr = name.rsplit("_", 1)
        time = pd.to_datetime(tstr, format="%Y%m%dT%H%M%S")

        # detect source type
        if "salt_geotiff_ScenM0" in folder:
            source = "Baseline"
            scenario_name = "Baseline"

        elif "temp_geotiff_ScenM0" in folder:
            source = "Baseline"
            scenario_name = "Baseline"

        else:
            source = "Scenario"

            # everything after "..._geotiff_"
            scenario_name = folder.split("_geotiff_")[-1]

        records.append(
            {
                "file": f,
                "variable": var,
                "time": time,
                "source": source,
                "scenario": scenario_name,      # <-- NEW
            }
        )

    except Exception:
        continue
if not records:
    st.error("No valid tif files found.")
    st.stop()

df_files = pd.DataFrame(records).sort_values("time")

#variables = sorted(df_files["variable"].unique())


#with st.sidebar:
 #   st.header("Inputs")
#
 #   selected_var = st.selectbox("Variable", variables)
  #  scenario = st.sidebar.selectbox("Select Scenario", list(geojson_files.keys()))
# --- sidebar FIRST ---
with st.sidebar:
    st.header("Inputs")

    source_options = sorted(df_files["source"].unique())

    selected_source = st.selectbox(
        "Dataset",
        source_options
    )
   
    variables = sorted(
        df_files[df_files["source"] == selected_source]["variable"].unique()
    )
    
    selected_var = st.selectbox("Variable", variables)

    scenario = st.selectbox(
        "Select Scenario",
        [None] + list(geojson_files.keys()),
        format_func=lambda x: "None" if x is None else x,
        index=0,
    )

    if scenario is not None:
        if scenario not in st.session_state.selected_scenarios:
            st.session_state.selected_scenarios.append(scenario)
        
    
    #scenario = st.multiselect(
     #   "Select Scenarios",
      #  list(geojson_files.keys())
    #)

#    folder_scenario = SCENARIOS.get(scenario)
    if selected_source == "Baseline":

    # Physics reference
        df_var = df_files[
            (df_files["variable"] == selected_var) &
            (df_files["source"] == "Baseline")
        ]

        folder_scenario = "Baseline"

    else:
    # Biology GeoTIFFs
        if selected_source == "Baseline":

            folder_scenario = "Baseline"

        else:

            if scenario is None:
                folder_scenario = "ScenM0"     # default until user picks one
            else:
                folder_scenario = SCENARIO_TO_FOLDER[scenario]

       
            df_var = df_files[
                (df_files["variable"] == selected_var) &
                (df_files["source"] == "Scenario") &
                (df_files["scenario"] == folder_scenario)
            ]

        
    if df_var.empty:
        st.error(
            f"No GeoTIFFs found for:\n"
            f"Variable = {selected_var}\n"
            f"Source = {selected_source}\n"
            f"Scenario = {folder_scenario}"
        )
        st.stop()

# Metadata for currently displayed raster
    scenario_name = "Baseline" if selected_source == "Baseline" else folder_scenario


# Ensure time column is datetime
    df_var = df_var.copy()
    df_var["time"] = pd.to_datetime(df_var["time"])

    times = (
        df_var["time"]
        .sort_values()
        .reset_index(drop=True)
    )

    selected_time = st.select_slider(
        "Time",
        options=times,
        value=times.iloc[0]
    )

    idx = (
        df_var["time"] - pd.Timestamp(selected_time)
    ).abs().idxmin()

    selected_row = df_var.loc[idx]
    selected_tif = selected_row["file"]



    st.write("Selected file:", selected_tif.name)

    
    opacity = st.slider("TIFF overlay opacity", 0.0, 1.0, 0.75, 0.05)
    show_point_markers = st.checkbox("Show GeoJSON point markers", value=True)
    max_markers = st.slider("Max markers on map", 100, 5000, 1200, 100)

if "last_scenario" not in st.session_state:
    st.session_state.last_scenario = None
    st.session_state.force_fitbounds = False

if scenario != st.session_state.last_scenario:
    st.session_state.last_scenario = scenario
    st.session_state.force_fitbounds = True
    
                 
if st.button("Clear scenarios"):
    st.session_state.selected_scenarios = []
    
SCENARIO_COLORS = {
    "Scenario 1": "#1f77b4",  # blue
    "Scenario 2": "#d62728",  # red
    "Scenario N": "#2ca02c",  # green
    "Scenario E": "#9467bd",  # purple
    "Scenario S": "#ff7f0e",  # orange
    "Scenario W": "#8c564b",  # brown
}

# defaults (no scenario selected)
# defaults
ts_df = pd.DataFrame(columns=["time", "value", "lon", "lat"])
point_df = pd.DataFrame(columns=["lon", "lat", "value_latest", "time_latest"])
avg_ts = pd.DataFrame(columns=["time", "value"])

if selected_source == "Scenario":

    if scenario is None:
        st.info("Select a scenario to display GeoJSON points and time series.")

    else:
        geojson_path = geojson_files[scenario]

        if not geojson_path.exists():
            st.error(f"GeoJSON file not found: {geojson_path}")

        else:
            ts_df = _cached_load_geojson_timeseries(str(geojson_path))

            if not ts_df.empty:
                ts_df["time"] = pd.to_datetime(ts_df["time"])
                selected_time_pd = pd.Timestamp(selected_time)

                ts_df_current = ts_df[
                    ts_df["time"] <= selected_time
                ].copy()

                avg_ts = (
                    ts_df_current.groupby("time", as_index=False)["value"]
                    .mean()
                    .sort_values("time")
                )


                point_df = (
                    ts_df_current
                    .sort_values("time")
                    .groupby(["lon", "lat"], as_index=False)
                    .agg(
                        value_latest=("value", "last"),
                        time_latest=("time", "last"),
                    )
                )


# --- MULTI-SCENARIO AGGREGATION (NEW) ---
all_avg_ts = {}

for scen in st.session_state.selected_scenarios:

    geojson_path = geojson_files[scen]

    if not geojson_path.exists():
        continue

    ts_df_tmp = _cached_load_geojson_timeseries(str(geojson_path))

    if ts_df_tmp.empty:
        continue

    ts_df_tmp["time"] = pd.to_datetime(ts_df_tmp["time"])
    ts_df_tmp["value"] = pd.to_numeric(ts_df_tmp["value"], errors="coerce")

    avg_tmp = (
        ts_df_tmp
        .groupby("time", as_index=False)["value"]
        .mean()
        .sort_values("time")
    )

    # Number of monopiles / farm locations
    N_FARMS = (
        ts_df_tmp[["lon", "lat"]]
        .drop_duplicates()
        .shape[0]
    )

    # Estimated harvest
    avg_tmp["harvest"] = avg_tmp["value"] * N_FARMS

    all_avg_ts[scen] = avg_tmp

@st.cache_data(show_spinner=False)
def _load_tif_cached(path_str: str, turbine_df):
    path = Path(path_str)
    tif_bytes = path.read_bytes()
    return _render_preview_from_tif(tif_bytes, turbine_df, max_size=1024)
    
#def _render_diff_rgba(diff: np.ndarray) -> np.ndarray:
def _render_diff_rgba(
    diff: np.ndarray,
    transform,
    turbine_df: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """
    Render a signed difference raster with red-blue diverging colormap.
    Positive = red, negative = blue, near zero = white/grey.
    """

    buffer_deg = 0.05

    min_lon = turbine_df["lon"].min() - buffer_deg
    max_lon = turbine_df["lon"].max() + buffer_deg

    min_lat = turbine_df["lat"].min() - buffer_deg
    max_lat = turbine_df["lat"].max() + buffer_deg

    # Convert lon/lat -> raster rows/cols
    r0, c0 = rowcol(transform, min_lon, max_lat)
    r1, c1 = rowcol(transform, max_lon, min_lat)

    r0, r1 = sorted((r0, r1))
    c0, c1 = sorted((c0, c1))

    # Keep indices inside raster
    r0 = max(r0, 0)
    c0 = max(c0, 0)

    r1 = min(r1, diff.shape[0])
    c1 = min(c1, diff.shape[1])

    # Mask invalid pixels
    mask = ~np.isfinite(diff)

    # Keep original diff for statistics
    window = diff[r0:r1, c0:c1]

    # --------------------------------------------------
    # Compute colour scale only over OWF window
    # --------------------------------------------------

    valid = np.abs(window[np.isfinite(window)])
    nonzero = valid[valid > 0]

    if nonzero.size > 0:
        # Normal case: real OWF differences exist
        v = np.nanpercentile(nonzero, 98)

    elif valid.size > 0:
        # OWF exists but no detectable difference
        v = 0.001

    else:
        # No valid OWF pixels: fallback to whole raster
        valid_all = np.abs(diff[np.isfinite(diff)])
        valid_all = valid_all[valid_all > 0]

        if valid_all.size > 0:
            v = np.nanpercentile(valid_all, 98)
        else:
            v = 0.001

    # Avoid numerical problems
    if not np.isfinite(v) or v == 0:
        v = 0.001

    # --------------------------------------------------
    # Render
    # --------------------------------------------------

    norm = mcolors.TwoSlopeNorm(
        vmin=-v,
        vcenter=0.0,
        vmax=v,
    )

    cmap = matplotlib.colormaps.get_cmap("RdBu_r")
    rgba = cmap(norm(np.where(mask, 0, diff)))

    rgba = (rgba * 255).astype(np.uint8)

    # Transparent invalid pixels
    rgba[mask, 3] = 0

    return rgba, v

if "current_tif" not in st.session_state:
    st.session_state.current_tif = None
    st.session_state.rgba = None
    st.session_state.bounds = None
    st.session_state.vmin = None
    st.session_state.vmax = None

try:
    tif_path = str(selected_tif)

    # Only reload if file actually changed
    if st.session_state.current_tif != tif_path:
        rgba, bounds, _, vmin, vmax = _load_tif_cached(tif_path, turbine_df)

        st.session_state.current_tif = tif_path
        st.session_state.rgba = rgba
        st.session_state.bounds = bounds
        st.session_state.vmin = vmin
        st.session_state.vmax = vmax

    # reuse cached values
    rgba = st.session_state.rgba
    south, west, north, east = st.session_state.bounds
    vmin = st.session_state.vmin
    vmax = st.session_state.vmax

except Exception as e:
    st.error(f"Failed to read/render local GeoTIFF: {e}")
    st.stop()

      
point_plot_df = point_df.copy()

if len(point_df) > max_markers:
    # Downsample markers for folium rendering performance; keep full point_df for nearest-neighbor logic.
    step = int(np.ceil(len(point_df) / max_markers))
    point_plot_df = point_df.iloc[::step].copy()

if point_df.empty:
    center = [(south + north) / 2, (west + east) / 2]
else:
    center = [float(point_df["lat"].mean()), float(point_df["lon"].mean())]
    
if "map_center" not in st.session_state:
    st.session_state.map_center = None

if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 9

# --- compute map bounds from actual data ---
if not point_df.empty:
    point_bounds = [
        [float(point_df["lat"].min()), float(point_df["lon"].min())],
        [float(point_df["lat"].max()), float(point_df["lon"].max())],
    ]
else:
    point_bounds = [[south, west], [north, east]]


# --- build map ---
#m = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap", control_scale=True)

m = folium.Map(
    location=[0, 0],
    zoom_start=2,
    tiles="OpenStreetMap",
    control_scale=True,
)

# compute final bounds ONLY once
if scenario is not None and not point_df.empty:
    final_bounds = [
        [float(point_df["lat"].min()), float(point_df["lon"].min())],
        [float(point_df["lat"].max()), float(point_df["lon"].max())],
    ]
else:
    final_bounds = [[south, west], [north, east]]

folium.FitBounds(final_bounds).add_to(m)


#folium.FitBounds([[south, west], [north, east]]).add_to(m)

 
folium.raster_layers.ImageOverlay(
    image=_rgba_to_data_url(rgba),
    bounds=[[south, west], [north, east]],
    opacity=opacity,
    interactive=True,
    cross_origin=False,
    zindex=1,
).add_to(m)

# --- markers FIRST ---
if show_point_markers:
    for row in point_plot_df.itertuples(index=False):
        popup = folium.Popup(
            f"lon={row.lon:.6f}<br>lat={row.lat:.6f}<br>"
            f"value={row.value_latest:.4f}<br>time={row.time_latest}",
            max_width=260,
        )

        folium.CircleMarker(
            location=[float(row.lat), float(row.lon)],
            radius=3,
            color="#ffffff",
            weight=1,
            fill=True,
            fill_color="#ff006e",
            fill_opacity=0.9,
            popup=popup,
        ).add_to(m)

      
# --- turbine markers ---
for row in turbine_df.itertuples(index=False):

    folium.Marker(
        location=[row.lat, row.lon],
        icon=folium.DivIcon(
            html="""
            <div style="
                font-size:18px;
                color:black;
                line-height:18px;
                text-align:center;">
                +
            </div>
            """
        )
    ).add_to(m)
    
colors = [
    mcolors.to_hex(c)
    for c in matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))
]

colormap = cm.LinearColormap(
    colors=colors,
    vmin=vmin,
    vmax=vmax
)
#colormap.caption = selected_var
info = VARIABLE_INFO.get(
    selected_var,
    {"name": selected_var, "unit": "-"}
)

colormap.caption = f"{info['name']} ({info['unit']})"
colormap.add_to(m)
        

# --- layer control / bounds LAST ---
folium.LayerControl().add_to(m)

# ONLY apply FitBounds once (initial load)
    
if "selected_points" not in st.session_state:
    st.session_state.selected_points = []

map_state = st_folium(m, use_container_width=True, height=700)
if map_state is None:
    map_state = {}
    #st.stop()

#st.success("Reached info section")

legend_col, info_col = st.columns([1, 2])

with legend_col:

    st.markdown("### Map information")

    st.write("✚ Offshore wind turbine foundation")
    st.write("🔴 Low-trophic aquaculture unit")
    st.write("🌈 Environmental variable (GeoTIFF)")
    st.write("Color bar = value range")


with info_col:

    info = VARIABLE_INFO.get(
        selected_var.lower(),
        {
            "name": selected_var,
            "unit": "-",
            "description": ""
        }
    )

    st.markdown(f"### {info['name']}")

    st.markdown(
        f"**Variable:** `{selected_var}` — {info['name']}  \n"
        f"**Unit:** {info['unit']}"
    )
    
    st.write(info["description"])

    st.divider()

    st.markdown("### Current map")

    st.write(f"**Dataset:** {selected_source}")

    if selected_source == "Baseline":
        st.write("**Scenario:** Baseline")
    else:
        st.write(f"**Scenario:** {folder_scenario}")

    st.write(f"**Variable:** {selected_var}")

    st.write(f"**Date:** {selected_row['time'].strftime('%Y-%m-%d')}")


# --------------------------------------------------
# Multi-scenario comparison
# --------------------------------------------------

st.subheader("Average mussel biomass – Multi-scenario comparison")

selected_time_pd = pd.to_datetime(selected_time)

col1, col2 = st.columns(2)

# --------------------------------------------------
# LEFT: time series (from t0 until selected time)
# --------------------------------------------------

with col1:

    st.markdown("**Average biomass [kg/ind]**")

    plot_df = pd.DataFrame()

    for scen, df in all_avg_ts.items():

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])

        # keep only data up to selected time
        df = df[df["time"] <= selected_time_pd]

        if df.empty:
            continue

        ts = (
            df
            .set_index("time")[["value"]]
            .rename(columns={"value": scen})
        )

        plot_df = ts if plot_df.empty else plot_df.join(ts, how="outer")

    import altair as alt

    if plot_df.empty:

        st.warning("No scenario data available for the selected time.")

    else:

        plot_long = (
            plot_df
            .reset_index(names="time")
            .melt(
                id_vars="time",
                var_name="scenario",
                value_name="value",
            )
        )

        selected_scenarios = (
            plot_long["scenario"]
            .dropna()
            .unique()
            .tolist()
        )

        chart = alt.Chart(plot_long).mark_line().encode(

            x=alt.X("time:T", title="Time"),

            y=alt.Y(
                "value:Q",
                scale=alt.Scale(type="log"),
                title="Average biomass [kg individual⁻¹]",
            ),

            color=alt.Color(
                "scenario:N",
                scale=alt.Scale(
                    domain=selected_scenarios,
                    range=[
                        SCENARIO_COLORS.get(s, "#808080")
                        for s in selected_scenarios
                    ],
                ),
                legend=alt.Legend(title="Scenario"),
            ),
        )

        st.altair_chart(chart, use_container_width=True)

# --------------------------------------------------
# RIGHT: harvest at selected time
# --------------------------------------------------

with col2:

    st.markdown("**Estimated mussel harvest [kg]**")

    bar_data = {}

    for scen, df in all_avg_ts.items():

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])

        # keep only data up to selected time
        df = df[df["time"] <= selected_time_pd]

        if df.empty:
            continue

        # take the latest available row up to selected time
        row = df.sort_values("time").iloc[-1]

        bar_data[scen] = row["harvest"]

    if not bar_data:

        st.warning("No harvest data available for the selected time.")

    else:

        bar_df = pd.DataFrame.from_dict(
            bar_data,
            orient="index",
            columns=["Harvest"],
        )

        bar_long = (
            bar_df
            .reset_index()
            .rename(columns={"index": "scenario"})
        )

        selected_scenarios = list(bar_data.keys())

        chart = alt.Chart(bar_long).mark_bar().encode(

            x=alt.X("scenario:N", title="Scenario"),

            y=alt.Y(
                "Harvest:Q",
                title="Estimated harvest biomass [kg]",
            ),

            color=alt.Color(
                "scenario:N",
                scale=alt.Scale(
                    domain=selected_scenarios,
                    range=[
                        SCENARIO_COLORS.get(s, "#808080")
                        for s in selected_scenarios
                    ],
                ),
                legend=alt.Legend(title="Scenario"),
            ),
        )

        st.altair_chart(chart, use_container_width=True)

    # =========================
# GEO-TIFF DIFFERENCE MAP
# =========================

st.subheader("Scenario − ScenM0 (biology baseline) difference")

if selected_source != "Scenario":
    st.info("Switch to the Scenario dataset to compute differences.")

elif scenario is None:
    st.info("Select a scenario to compute differences.")

else:
    try:
        # --------------------------------------------------
        # Current scenario GeoTIFF
        # --------------------------------------------------

        scen_path = Path(selected_tif)
        scen_bytes = scen_path.read_bytes()

        scen_rgba, scen_bounds, _, _, _ = _load_tif_cached(str(scen_path), turbine_df)

        scenario_time = pd.Timestamp(selected_row["time"])

        # --------------------------------------------------
        # Reference GeoTIFF = ScenM0
        # --------------------------------------------------

        ref_df = df_files[
            (df_files["variable"] == selected_var) &
            (df_files["source"] == "Scenario") &
            (df_files["scenario"] == "ScenM0")
        ].copy()

        if ref_df.empty:
            st.error(
                f"No ScenM0 reference GeoTIFF found.\n"
                f"Variable = {selected_var}"
            )
            st.stop()

        # --------------------------------------------------
        # Match nearest timestamp
        # --------------------------------------------------

        ref_df["time"] = pd.to_datetime(ref_df["time"])

        idx_ref = (ref_df["time"] - scenario_time).abs().idxmin()

        ref_row = ref_df.loc[idx_ref]

        ref_file = ref_row["file"]
        ref_bytes = ref_file.read_bytes()

        # ---------- TEMPORARY DEBUG ----------
        st.write(f"Scenario TIFF : {Path(selected_tif).name}")
        st.write(f"Reference TIFF: {Path(ref_file).name}")
        st.write(f"Scenario time : {scenario_time}")
        st.write(f"Reference time: {ref_row['time']}")

        # --------------------------------------------------
        # Read + align rasters
        # --------------------------------------------------

        with MemoryFile(scen_bytes) as mem_s:
            with mem_s.open() as src_s:

                with MemoryFile(ref_bytes) as mem_r:
                    with mem_r.open() as src_r:

                        #scen = _clean(src_s.read(1), src_s)
                        #ref = _clean(src_r.read(1), src_r)
                        scen = _clean(src_s.read(1), src_s, selected_var)
                        ref  = _clean(src_r.read(1), src_r, selected_var)
                        ref_aligned = np.full_like(
                            scen,
                            np.nan,
                            dtype=np.float32,
                        )

                        reproject(
                            source=ref,
                            destination=ref_aligned,
                            src_transform=src_r.transform,
                            src_crs=src_r.crs,
                            dst_transform=src_s.transform,
                            dst_crs=src_s.crs,
                            resampling=Resampling.bilinear,
                        )
                        
                        transform = src_s.transform

        # --------------------------------------------------
        # Compute difference
        # --------------------------------------------------

        diff = scen - ref_aligned

        valid = np.isfinite(diff)

        diff_smooth = diff.copy()
        diff_smooth[~valid] = 0

        diff_smooth = gaussian_filter(
            diff_smooth,
            sigma=1.0,
        )

        diff_smooth[~valid] = np.nan

        diff_rgba, v = _render_diff_rgba(
            diff_smooth,
            src_s.transform,
            turbine_df,
        )

        south, west, north, east = scen_bounds

        diff_map = folium.Map(
            tiles="OpenStreetMap",
            control_scale=True,
        )

        folium.FitBounds(final_bounds).add_to(diff_map)

        folium.raster_layers.ImageOverlay(
            image=_rgba_to_data_url(diff_rgba),
            bounds=[[south, west], [north, east]],
            opacity=0.75,
        ).add_to(diff_map)

        diff_colormap = cm.LinearColormap(
            colors=[
                "#2166ac",
                "#67a9cf",
                "#f7f7f7",
                "#ef8a62",
                "#b2182b",
            ],
            vmin=-v,
            vmax=v,
        )

        
        diff_colormap.caption = (
            f"{info['name']} difference ({folder_scenario} − ScenM0) [{info['unit']}]"
        )
        diff_colormap.add_to(diff_map)

        folium.LayerControl().add_to(diff_map)

        st_folium(
            diff_map,
            use_container_width=True,
            height=550,
        )

        st.caption("🔴 Higher than ScenM0   |   🔵 Lower than ScenM0")

    except Exception as e:
        st.warning(f"Could not compute GeoTIFF difference: {e}")    

if ts_df.empty:
    st.warning("No valid `value` and `time` entries found in GeoJSON.")
else:
    st.caption(
        f"GeoJSON records: {len(ts_df):,} | unique locations: {len(point_df):,} | "
        f"markers rendered: {len(point_plot_df):,}"
    )
    clicked_lat, clicked_lon = None, None
    if isinstance(map_state, dict):
        obj_click = map_state.get("last_object_clicked")
        if isinstance(obj_click, dict) and "lat" in obj_click and "lng" in obj_click:
            clicked_lat = float(obj_click["lat"])
            clicked_lon = float(obj_click["lng"])
        elif isinstance(map_state.get("last_clicked"), dict):
            clicked_lat = float(map_state["last_clicked"].get("lat"))
            clicked_lon = float(map_state["last_clicked"].get("lng"))

    if clicked_lat is not None and clicked_lon is not None and not point_df.empty:
        nearest = _select_nearest_point(point_df, clicked_lon, clicked_lat)

        selected_lon = float(nearest["lon"])
        selected_lat = float(nearest["lat"])

        point_id = (round(selected_lon, 6), round(selected_lat, 6))

        if "selected_points" not in st.session_state:
            st.session_state.selected_points = []

        if point_id not in st.session_state.selected_points:
            st.session_state.selected_points.append(point_id)

            st.write("Added point:", point_id)

    elif not point_df.empty:
        selected_lon = float(point_df.iloc[0]["lon"])
        selected_lat = float(point_df.iloc[0]["lat"])

        st.caption(
            f"No map click yet - showing first point: "
            f"lon={selected_lon:.6f}, lat={selected_lat:.6f}"
        )

    else:
        st.warning("No point data available.")
        st.stop()
    point_ts = ts_df_current[
        (np.isclose(ts_df_current["lon"], selected_lon)) &
        (np.isclose(ts_df_current["lat"], selected_lat))
    ].copy()


    #bad = point_ts[~point_ts["time"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")]
    #st.write("Bad format:", bad["time"].unique())
# Also catch invalid dates like month=00
    #parsed = pd.to_datetime(point_ts["time"], errors="coerce")
    #bad2 = point_ts[parsed.isna()]
    #st.write("Unparseable values:", bad2["time"].unique())
    point_ts["time"] = pd.to_datetime(point_ts["time"])
    point_ts = point_ts.sort_values("time")
    time_num = pd.to_numeric(point_ts["time"], errors="coerce")
    if time_num.notna().all():
        point_ts = point_ts.assign(_time_num=time_num).sort_values("_time_num").drop(columns=["_time_num"])
    plot_df = pd.DataFrame()

    for lon_sel, lat_sel in st.session_state.selected_points:

        tol = 1e-5

        ts = ts_df_current[
            (np.abs(ts_df_current["lon"] - lon_sel) < tol) &
            (np.abs(ts_df_current["lat"] - lat_sel) < tol)
        ].copy()

        if ts.empty:
            continue

        ts = ts.sort_values("time")

        label = f"{lon_sel:.3f},{lat_sel:.3f}"

        ts = (
            ts.set_index("time")[["value"]]
              .rename(columns={"value": label})
        )

        plot_df = ts if plot_df.empty else plot_df.join(ts, how="outer")
    
    #chart_container.warning("No time series found for selected points")
    chart_container = st.empty()

    if plot_df.empty:

        missing_points = len(st.session_state.selected_points)

#        chart_container.warning(
 #           f"No time series found for {missing_points} selected point(s)"
  #      )

    else:
        chart_container.line_chart(plot_df, height=300)

        st.dataframe(
            point_ts[["time", "value", "lon", "lat"]],
            use_container_width=True
        )
    if st.button("Clear selected points"):
        st.session_state.selected_points = []
