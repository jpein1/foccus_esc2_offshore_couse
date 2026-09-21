import io
import json
import os
from urllib.request import Request, urlopen

import altair as alt
import folium
import numpy as np
import pandas as pd
import rasterio
import streamlit as st

from folium.plugins import Fullscreen
from matplotlib import cm
from matplotlib.colors import Normalize
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.warp import reproject
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="OWF & LTA co-use dashboard",
    page_icon="🌊",
    layout="wide",
)

st.title("OWF & LTA co-use dashboard")
st.caption("Mussel biomass, scenarios, and spatial analysis viewer")


# ============================================================
# PUBLIC DATA CONFIGURATION
# ============================================================
#
# Data are publicly readable through HTTPS.
#
# IMPORTANT:
# The application does NOT use S3 ListObjectsV2.
#
# A manifest.json file must exist at:
#
# https://minio.dive.edito.eu/project-foccus/
#   Hereon/ESC2_blue_economy/manifest.json
#
# The manifest contains relative paths such as:
#
# {
#   "geotiff": [
#     "geotiff/ScenM0/salt_20200501T000000.tif",
#     "geotiff/ScenM0/temp_20200501T000000.tif",
#     "geotiff/ScenM2/xxx_20200501T000000.tif",
#     "geotiff/ScenM3/xxx_20200501T000000.tif"
#   ],
#   "geojson": [
#     "geojson/harvest_timeseries_scenario_Scen_M2.geojson",
#     "geojson/harvest_timeseries_scenario_Scen_M3.geojson"
#   ],
#   "csv": [
#     "Meerwind_monopiles_lonlat.csv"
#   ]
# }
#
# ============================================================

PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://minio.dive.edito.eu/project-foccus/Hereon/ESC2_blue_economy",
).rstrip("/")

PUBLIC_GEOTIFF_URL = f"{PUBLIC_BASE_URL}/geotiff"
PUBLIC_GEOJSON_URL = f"{PUBLIC_BASE_URL}/geojson"
PUBLIC_MANIFEST_URL = f"{PUBLIC_BASE_URL}/manifest.json"


# ============================================================
# DATA STRUCTURE
# ============================================================
#
# project-foccus/
# └── Hereon/
#     └── ESC2_blue_economy/
#         ├── manifest.json
#         ├── Meerwind_monopiles_lonlat.csv
#         ├── geotiff/
#         │   ├── ScenM0/
#         │   │   ├── salt_YYYYMMDDTHHMMSS.tif
#         │   │   ├── temp_YYYYMMDDTHHMMSS.tif
#         │   │   └── ...
#         │   ├── ScenM2/
#         │   │   └── <scenario variables>_YYYYMMDDTHHMMSS.tif
#         │   └── ScenM3/
#         │       └── <scenario variables>_YYYYMMDDTHHMMSS.tif
#         └── geojson/
#             ├── harvest_timeseries_scenario_Scen_M2.geojson
#             └── harvest_timeseries_scenario_Scen_M3.geojson
#
# IMPORTANT:
#   salt and temp are BASELINE ONLY.
#   They must ONLY ever come from ScenM0.
# ============================================================


REFERENCE_FOLDER = "ScenM0"

SCENARIO_TO_FOLDER = {
    "Scenario 1": "ScenM2",
    "Scenario 2": "ScenM3",
}

BASELINE_ONLY_VARIABLES = {
    "salt",
    "temp",
}


# ============================================================
# PUBLIC HTTP HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def _read_public_bytes(url):
    """
    Download a publicly accessible object over HTTPS.

    No S3 credentials are required.
    """

    request = Request(
        url,
        headers={
            "User-Agent": "OWF-LTA-co-use-dashboard/1.0",
        },
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:
        return response.read()


def _public_object_url(key):
    """
    Convert a manifest-relative object path into
    its public HTTPS URL.
    """

    return (
        f"{PUBLIC_BASE_URL}/"
        f"{key.lstrip('/')}"
    )


# ============================================================
# LOAD PUBLIC MANIFEST
# ============================================================

@st.cache_data(show_spinner=False)
def _load_manifest():
    """
    Load the public project manifest.

    The manifest replaces S3 ListObjectsV2 discovery.
    """

    manifest_bytes = _read_public_bytes(
        PUBLIC_MANIFEST_URL
    )

    return json.loads(
        manifest_bytes.decode("utf-8")
    )


try:
    manifest = _load_manifest()

    geotiff_keys = manifest.get(
        "geotiff",
        [],
    )

    geojson_keys = manifest.get(
        "geojson",
        [],
    )

    csv_keys = manifest.get(
        "csv",
        [],
    )

except Exception as exc:
    st.error(
        "Could not load the public project manifest.\n\n"
        f"{exc}"
    )
    st.stop()


# ============================================================
# DISCOVER GEOTIFF FILES FROM MANIFEST
# ============================================================

records = []

for key in geotiff_keys:

    if not isinstance(key, str):
        continue

    key = key.strip("/")

    if not key.lower().endswith(
        (".tif", ".tiff")
    ):
        continue

    parts = key.split("/")

    # Expected:
    # geotiff/ScenM0/file.tif
    # geotiff/ScenM2/file.tif
    # geotiff/ScenM3/file.tif

    if len(parts) < 2:
        continue

    scenario_folder = parts[-2]
    filename = parts[-1]

    # Only accept folders that are part of the known structure.
    if scenario_folder not in {
        "ScenM0",
        "ScenM2",
        "ScenM3",
    }:
        continue

    # Expected:
    # variable_YYYYMMDDTHHMMSS.tif

    try:
        variable, timestamp_string = filename.rsplit(
            "_",
            1,
        )

        timestamp_string = os.path.splitext(
            timestamp_string
        )[0]

        timestamp = pd.to_datetime(
            timestamp_string,
            format="%Y%m%dT%H%M%S",
        )

    except Exception:
        # Ignore files that do not follow the expected naming scheme.
        continue

    variable = variable.strip()

    if not variable:
        continue

    # --------------------------------------------------------
    # CRITICAL:
    #
    # salt and temp are baseline/reference variables.
    #
    # Explicitly reject them from scenario folders.
    # --------------------------------------------------------

    if variable in BASELINE_ONLY_VARIABLES:
        if scenario_folder != REFERENCE_FOLDER:
            continue

    records.append(
        {
            "file": key,
            "filename": filename,
            "variable": variable,
            "time": timestamp,
            "source": (
                "Baseline"
                if scenario_folder == REFERENCE_FOLDER
                else "Scenario"
            ),
            "scenario": scenario_folder,
        }
    )


df_files = pd.DataFrame(records)

if df_files.empty:
    st.error(
        "No valid GeoTIFF files were found in the public manifest."
    )
    st.stop()

df_files = df_files.sort_values(
    [
        "scenario",
        "variable",
        "time",
    ]
).reset_index(drop=True)


# ============================================================
# GEOJSON DISCOVERY FROM MANIFEST
# ============================================================

geojson_objects_by_name = {}

for key in geojson_keys:

    if not isinstance(key, str):
        continue

    key = key.strip("/")

    filename = key.split("/")[-1]

    if filename.lower().endswith(".geojson"):
        geojson_objects_by_name[
            filename.lower()
        ] = key


def _find_geojson_key(*patterns):
    """
    Find a GeoJSON whose filename contains all supplied patterns.
    """

    patterns = [
        pattern.lower()
        for pattern in patterns
    ]

    for filename_lower, key in geojson_objects_by_name.items():

        if all(
            pattern in filename_lower
            for pattern in patterns
        ):
            return key

    return None


geojson_files = {}

scenario_1_geojson = _find_geojson_key(
    "harvest_timeseries",
    "scen_m2",
)

scenario_2_geojson = _find_geojson_key(
    "harvest_timeseries",
    "scen_m3",
)

if scenario_1_geojson:
    geojson_files["Scenario 1"] = scenario_1_geojson

if scenario_2_geojson:
    geojson_files["Scenario 2"] = scenario_2_geojson


# ============================================================
# GEOJSON LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def _load_geojson_timeseries_from_bytes(geojson_bytes):
    """
    Load harvest/biomass time series from GeoJSON bytes.
    """

    data = json.loads(
        geojson_bytes.decode("utf-8")
    )

    rows = []

    for feature in data.get(
        "features",
        [],
    ):

        properties = feature.get(
            "properties",
            {},
        )

        geometry = feature.get(
            "geometry",
            {},
        )

        coordinates = geometry.get(
            "coordinates"
        )

        if not coordinates:
            continue

        if len(coordinates) < 2:
            continue

        lon = coordinates[0]
        lat = coordinates[1]

        value = properties.get("bwmus")
        time_value = properties.get("time")

        if value is None or time_value is None:
            continue

        try:
            time = pd.to_datetime(
                time_value
            )

            value = float(value)
            lon = float(lon)
            lat = float(lat)

        except Exception:
            continue

        rows.append(
            {
                "time": time,
                "value": value,
                "lon": lon,
                "lat": lat,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _cached_load_geojson_timeseries(key):
    geojson_bytes = _read_public_bytes(
        _public_object_url(key)
    )

    return _load_geojson_timeseries_from_bytes(
        geojson_bytes
    )


# ============================================================
# TURBINE CSV
# ============================================================

turbine_key = None

for key in csv_keys:

    if not isinstance(key, str):
        continue

    if (
        key.split("/")[-1].lower()
        == "meerwind_monopiles_lonlat.csv"
    ):
        turbine_key = key.strip("/")
        break


@st.cache_data(show_spinner=False)
def _load_turbines(key):

    csv_bytes = _read_public_bytes(
        _public_object_url(key)
    )

    try:
        df = pd.read_csv(
            io.BytesIO(csv_bytes),
            header=None,
            sep=r"\s+",
            engine="python",
        )

    except Exception:
        df = pd.read_csv(
            io.BytesIO(csv_bytes),
            header=None,
        )

    if df.shape[1] < 2:
        return pd.DataFrame(
            columns=[
                "lon",
                "lat",
            ]
        )

    df = df.iloc[:, :2].copy()

    df.columns = [
        "lon",
        "lat",
    ]

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce",
    )

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "lon",
            "lat",
        ]
    )

    return df


if turbine_key:

    try:
        turbines = _load_turbines(
            turbine_key
        )

    except Exception:
        turbines = pd.DataFrame(
            columns=[
                "lon",
                "lat",
            ]
        )

else:

    turbines = pd.DataFrame(
        columns=[
            "lon",
            "lat",
        ]
    )


# ============================================================
# TIFF RENDERING
# ============================================================

def _render_preview_from_tif(
    tif_bytes,
    max_size=1024,
):
    """
    Render a GeoTIFF into an RGBA image.

    Returns:
        rgba, bounds, vmin, vmax
    """

    with MemoryFile(
        tif_bytes
    ) as memfile:

        with memfile.open() as src:

            if src.crs is None:
                raise ValueError(
                    "GeoTIFF has no CRS."
                )

            if src.crs.to_epsg() != 4326:
                raise ValueError(
                    f"GeoTIFF CRS is {src.crs}, "
                    "but EPSG:4326 is required."
                )

            width = src.width
            height = src.height

            scale = min(
                1.0,
                max_size / max(
                    width,
                    height,
                ),
            )

            out_width = max(
                1,
                int(width * scale),
            )

            out_height = max(
                1,
                int(height * scale),
            )

            if src.count >= 3:

                data = src.read(
                    [1, 2, 3],
                    out_shape=(
                        3,
                        out_height,
                        out_width,
                    ),
                    resampling=Resampling.bilinear,
                ).astype(
                    np.float32
                )

                data = np.nan_to_num(
                    data,
                    nan=0.0,
                    posinf=255.0,
                    neginf=0.0,
                )

                if data.max() <= 1.0:
                    data *= 255.0

                data = np.clip(
                    data,
                    0,
                    255,
                ).astype(
                    np.uint8
                )

                rgba = np.moveaxis(
                    data,
                    0,
                    -1,
                )

                alpha = np.full(
                    (
                        out_height,
                        out_width,
                    ),
                    255,
                    dtype=np.uint8,
                )

                rgba = np.dstack(
                    [
                        rgba,
                        alpha,
                    ]
                )

                vmin = None
                vmax = None

            else:

                data = src.read(
                    1,
                    out_shape=(
                        out_height,
                        out_width,
                    ),
                    resampling=Resampling.bilinear,
                ).astype(
                    np.float32
                )

                nodata = src.nodata

                valid = np.isfinite(
                    data
                )

                if nodata is not None:
                    valid &= (
                        data != nodata
                    )

                if not np.any(valid):
                    raise ValueError(
                        "GeoTIFF contains no valid data."
                    )

                values = data[valid]

                vmin = float(
                    np.nanpercentile(
                        values,
                        5,
                    )
                )

                vmax = float(
                    np.nanpercentile(
                        values,
                        95,
                    )
                )

                if (
                    not np.isfinite(vmin)
                    or not np.isfinite(vmax)
                    or vmin == vmax
                ):
                    vmin = float(
                        np.nanmin(values)
                    )

                    vmax = float(
                        np.nanmax(values)
                    )

                    if vmin == vmax:
                        vmax = vmin + 1.0

                clipped = np.clip(
                    data,
                    vmin,
                    vmax,
                )

                norm = Normalize(
                    vmin=vmin,
                    vmax=vmax,
                )

                cmap = cm.get_cmap(
                    "viridis"
                )

                rgba_float = cmap(
                    norm(clipped)
                )

                rgba = (
                    rgba_float * 255
                ).astype(
                    np.uint8
                )

                rgba[
                    ~valid,
                    3,
                ] = 0

            bounds = [
                [
                    src.bounds.bottom,
                    src.bounds.left,
                ],
                [
                    src.bounds.top,
                    src.bounds.right,
                ],
            ]

    return (
        rgba,
        bounds,
        vmin,
        vmax,
    )


@st.cache_data(show_spinner=False)
def _load_tif_cached(key):

    tif_bytes = _read_public_bytes(
        _public_object_url(key)
    )

    return _render_preview_from_tif(
        tif_bytes
    )


# ============================================================
# SCENARIO / REFERENCE HELPERS
# ============================================================

def _files_for_folder(folder):

    return df_files[
        df_files["scenario"] == folder
    ].copy()


def _variables_for_folder(folder):
    """
    Return variables actually available in a folder.

    Because salt/temp have already been filtered during discovery,
    this also guarantees they cannot appear for scenario folders.
    """

    folder_df = _files_for_folder(
        folder
    )

    if folder_df.empty:
        return []

    return sorted(
        folder_df["variable"]
        .dropna()
        .unique()
        .tolist()
    )


def _reference_files_for_variable(variable):
    """
    Return reference/baseline files.

    IMPORTANT:
    Always searches ONLY ScenM0.
    """

    return df_files[
        (df_files["scenario"] == REFERENCE_FOLDER)
        & (df_files["variable"] == variable)
    ].copy()


def _scenario_files_for_variable(
    scenario_folder,
    variable,
):
    """
    Return scenario files.

    salt and temp are explicitly blocked here as an
    additional safety check.
    """

    if variable in BASELINE_ONLY_VARIABLES:
        return pd.DataFrame(
            columns=df_files.columns
        )

    return df_files[
        (df_files["scenario"] == scenario_folder)
        & (df_files["variable"] == variable)
    ].copy()


# ============================================================
# SESSION STATE
# ============================================================

if "selected_scenarios" not in st.session_state:
    st.session_state.selected_scenarios = []

if "selected_points" not in st.session_state:
    st.session_state.selected_points = []

if "current_tif" not in st.session_state:
    st.session_state.current_tif = None

if "current_rgba" not in st.session_state:
    st.session_state.current_rgba = None

if "current_bounds" not in st.session_state:
    st.session_state.current_bounds = None

if "current_vmin" not in st.session_state:
    st.session_state.current_vmin = None

if "current_vmax" not in st.session_state:
    st.session_state.current_vmax = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Data selection")

selected_source = st.sidebar.radio(
    "Dataset",
    [
        "Baseline",
        "Scenario",
    ],
)


# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

if selected_source == "Baseline":

    active_scenario_folder = REFERENCE_FOLDER
    selected_scenario = None

    st.sidebar.info(
        "Baseline data are read from ScenM0."
    )


# ------------------------------------------------------------
# SCENARIO
# ------------------------------------------------------------

else:

    available_scenarios = [
        name
        for name, folder in SCENARIO_TO_FOLDER.items()
        if not _files_for_folder(folder).empty
    ]

    if not available_scenarios:

        st.sidebar.error(
            "No scenario GeoTIFF folders were found."
        )

        st.stop()

    selected_scenario = st.sidebar.selectbox(
        "Select Scenario",
        available_scenarios,
    )

    active_scenario_folder = (
        SCENARIO_TO_FOLDER[
            selected_scenario
        ]
    )

    if (
        selected_scenario
        not in st.session_state.selected_scenarios
    ):

        st.session_state.selected_scenarios.append(
            selected_scenario
        )


# ============================================================
# VARIABLE SELECTION
# ============================================================

available_variables = _variables_for_folder(
    active_scenario_folder
)

if not available_variables:

    st.error(
        f"No GeoTIFF variables were found in "
        f"{active_scenario_folder}."
    )

    st.stop()


# Extra explicit filtering:
#
# salt/temp may ONLY appear when ScenM0 is active.

if active_scenario_folder != REFERENCE_FOLDER:

    available_variables = [
        variable
        for variable in available_variables
        if variable not in BASELINE_ONLY_VARIABLES
    ]


if not available_variables:

    st.error(
        f"No scenario variables are available in "
        f"{active_scenario_folder}."
    )

    st.stop()


selected_var = st.sidebar.selectbox(
    "Variable",
    available_variables,
)


# ============================================================
# TIME SELECTION
# ============================================================

if selected_source == "Baseline":

    active_files = _files_for_folder(
        REFERENCE_FOLDER
    )

    active_files = active_files[
        active_files["variable"]
        == selected_var
    ].copy()

else:

    active_files = _scenario_files_for_variable(
        active_scenario_folder,
        selected_var,
    )


if active_files.empty:

    st.error(
        f"No files found for variable "
        f"`{selected_var}` in "
        f"`{active_scenario_folder}`."
    )

    st.stop()


available_times = (
    active_files["time"]
    .sort_values()
    .drop_duplicates()
    .tolist()
)

selected_time = st.sidebar.selectbox(
    "Time",
    available_times,
    index=len(available_times) - 1,
    format_func=lambda x: pd.Timestamp(
        x
    ).strftime(
        "%Y-%m-%d %H:%M"
    ),
)


# ============================================================
# SELECT TIFF
# ============================================================

matching_files = active_files[
    active_files["time"]
    == selected_time
]

if matching_files.empty:

    st.error(
        "No GeoTIFF matches the selected "
        "variable and time."
    )

    st.stop()


selected_row = matching_files.iloc[0]

selected_tif_key = selected_row["file"]


# ============================================================
# LOAD CURRENT TIFF
# ============================================================

try:

    (
        rgba,
        bounds,
        vmin,
        vmax,
    ) = _load_tif_cached(
        selected_tif_key
    )

except Exception as exc:

    st.error(
        "Could not read the selected GeoTIFF.\n\n"
        f"{exc}"
    )

    st.stop()


st.session_state.current_tif = (
    selected_tif_key
)

st.session_state.current_rgba = rgba
st.session_state.current_bounds = bounds
st.session_state.current_vmin = vmin
st.session_state.current_vmax = vmax


# ============================================================
# INFO
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Dataset",
        (
            "Baseline / ScenM0"
            if selected_source == "Baseline"
            else selected_scenario
        ),
    )

with col2:

    st.metric(
        "Variable",
        selected_var,
    )

with col3:

    st.metric(
        "Time",
        pd.Timestamp(
            selected_time
        ).strftime(
            "%Y-%m-%d %H:%M"
        ),
    )


# ============================================================
# MAIN MAP
# ============================================================

center_lat = np.mean(
    [
        bounds[0][0],
        bounds[1][0],
    ]
)

center_lon = np.mean(
    [
        bounds[0][1],
        bounds[1][1],
    ]
)

m = folium.Map(
    location=[
        center_lat,
        center_lon,
    ],
    zoom_start=9,
    control_scale=True,
)

Fullscreen().add_to(m)


# ------------------------------------------------------------
# GeoTIFF image overlay
# ------------------------------------------------------------

image = folium.raster_layers.ImageOverlay(
    image=rgba,
    bounds=bounds,
    opacity=0.75,
    interactive=True,
    cross_origin=False,
    zindex=1,
)

image.add_to(m)


# ------------------------------------------------------------
# Turbine markers
# ------------------------------------------------------------

for _, turbine in turbines.iterrows():

    folium.Marker(
        location=[
            turbine["lat"],
            turbine["lon"],
        ],
        icon=folium.DivIcon(
            html="""
            <div style="
                font-size:18px;
                font-weight:bold;
                color:black;
                text-align:center;
            ">+</div>
            """
        ),
        tooltip="Meerwind monopile",
    ).add_to(m)


# ------------------------------------------------------------
# GeoJSON biomass points
# ------------------------------------------------------------

active_geojson_df = pd.DataFrame()

if (
    selected_source == "Scenario"
    and selected_scenario in geojson_files
):

    try:

        active_geojson_df = (
            _cached_load_geojson_timeseries(
                geojson_files[
                    selected_scenario
                ],
            )
        )

    except Exception as exc:

        st.warning(
            "Could not load scenario GeoJSON:\n"
            f"{exc}"
        )


# ------------------------------------------------------------
# Add current-time GeoJSON points
# ------------------------------------------------------------

if not active_geojson_df.empty:

    current_points = (
        active_geojson_df[
            active_geojson_df["time"]
            == selected_time
        ]
        .copy()
    )

    # If exact timestamp is not present,
    # use the closest timestamp.

    if current_points.empty:

        nearest_time_index = (
            active_geojson_df["time"]
            .sub(selected_time)
            .abs()
            .idxmin()
        )

        nearest_time = (
            active_geojson_df.loc[
                nearest_time_index,
                "time",
            ]
        )

        current_points = (
            active_geojson_df[
                active_geojson_df["time"]
                == nearest_time
            ]
            .copy()
        )

    for _, point in current_points.iterrows():

        folium.CircleMarker(
            location=[
                point["lat"],
                point["lon"],
            ],
            radius=4,
            weight=1,
            fill=True,
            tooltip=(
                f"Biomass: "
                f"{point['value']:.3f}"
            ),
        ).add_to(m)


# ------------------------------------------------------------
# Colorbar for single-band data
# ------------------------------------------------------------

if vmin is not None and vmax is not None:

    gradient = np.linspace(
        0,
        1,
        256,
    )

    cmap = cm.get_cmap(
        "viridis"
    )

    colors = [
        "#{:02x}{:02x}{:02x}".format(
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255),
        )
        for color in cmap(gradient)
    ]

    gradient_css = ",".join(
        colors
    )

    colorbar_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 9999;
        background: white;
        padding: 10px;
        border: 1px solid #aaa;
        font-size: 12px;
    ">
        <div style="
            width: 180px;
            height: 14px;
            background: linear-gradient(
                to right,
                {gradient_css}
            );
        "></div>

        <div style="
            display:flex;
            justify-content:space-between;
        ">
            <span>{vmin:.3g}</span>
            <span>{vmax:.3g}</span>
        </div>
    </div>
    """

    m.get_root().html.add_child(
        folium.Element(
            colorbar_html
        )
    )


# ============================================================
# DISPLAY MAP
# ============================================================

map_data = st_folium(
    m,
    width=None,
    height=650,
    returned_objects=[
        "last_object_clicked",
        "last_clicked",
    ],
)


# ============================================================
# DIFFERENCE MAP
# ============================================================
#
# Only scenario data get a difference map.
#
# Scenario raster:
#     ScenM2 or ScenM3
#
# Reference raster:
#     ALWAYS ScenM0
#
# salt/temp:
#     baseline only -> no scenario difference is attempted.
# ============================================================

st.subheader(
    "Scenario − Reference"
)

if selected_source == "Baseline":

    st.info(
        "Difference maps are only calculated for a scenario "
        "selection. The current dataset is the ScenM0 baseline."
    )

elif selected_var in BASELINE_ONLY_VARIABLES:

    st.info(
        f"`{selected_var}` is a baseline/reference variable "
        "available only in ScenM0, so no scenario difference "
        "map is calculated."
    )

else:

    scenario_df = _scenario_files_for_variable(
        active_scenario_folder,
        selected_var,
    )

    reference_df = _reference_files_for_variable(
        selected_var
    )

    if scenario_df.empty:

        st.info(
            f"No `{selected_var}` files are available "
            f"in {active_scenario_folder}."
        )

    elif reference_df.empty:

        st.warning(
            f"No reference `{selected_var}` files are "
            "available in ScenM0."
        )

    else:

        # Find the scenario raster corresponding
        # to the selected time.

        scenario_match = scenario_df[
            scenario_df["time"]
            == selected_time
        ]

        if scenario_match.empty:

            st.info(
                "No scenario raster exists at the "
                "selected timestamp."
            )

        else:

            scenario_row = (
                scenario_match.iloc[0]
            )

            scenario_time = (
                scenario_row["time"]
            )

            # ------------------------------------------------
            # Reference is ALWAYS selected from ScenM0.
            # ------------------------------------------------

            idx_ref = (
                reference_df["time"]
                .sub(scenario_time)
                .abs()
                .idxmin()
            )

            reference_row = (
                reference_df.loc[idx_ref]
            )

            scenario_key = (
                scenario_row["file"]
            )

            reference_key = (
                reference_row["file"]
            )

            try:

                scenario_bytes = (
                    _read_public_bytes(
                        _public_object_url(
                            scenario_key
                        )
                    )
                )

                reference_bytes = (
                    _read_public_bytes(
                        _public_object_url(
                            reference_key
                        )
                    )
                )

                with MemoryFile(
                    scenario_bytes
                ) as scen_mem:

                    with scen_mem.open() as scen_src:

                        with MemoryFile(
                            reference_bytes
                        ) as ref_mem:

                            with ref_mem.open() as ref_src:

                                if (
                                    scen_src.count != 1
                                    or ref_src.count != 1
                                ):

                                    st.warning(
                                        "Difference maps currently "
                                        "require single-band GeoTIFFs."
                                    )

                                else:

                                    scenario_data = (
                                        scen_src.read(
                                            1
                                        ).astype(
                                            np.float32
                                        )
                                    )

                                    reference_data = (
                                        ref_src.read(
                                            1
                                        ).astype(
                                            np.float32
                                        )
                                    )

                                    reference_aligned = (
                                        np.full(
                                            scenario_data.shape,
                                            np.nan,
                                            dtype=np.float32,
                                        )
                                    )

                                    reproject(
                                        source=reference_data,
                                        destination=reference_aligned,
                                        src_transform=ref_src.transform,
                                        src_crs=ref_src.crs,
                                        dst_transform=scen_src.transform,
                                        dst_crs=scen_src.crs,
                                        resampling=Resampling.bilinear,
                                    )

                                    diff = (
                                        scenario_data
                                        - reference_aligned
                                    )

                                    if (
                                        scen_src.nodata
                                        is not None
                                    ):

                                        diff[
                                            scenario_data
                                            == scen_src.nodata
                                        ] = np.nan

                                    if (
                                        ref_src.nodata
                                        is not None
                                    ):

                                        reference_aligned[
                                            reference_aligned
                                            == ref_src.nodata
                                        ] = np.nan

                                    valid_diff = np.isfinite(
                                        diff
                                    )

                                    if np.any(
                                        valid_diff
                                    ):

                                        diff_values = diff[
                                            valid_diff
                                        ]

                                        diff_abs = np.nanpercentile(
                                            np.abs(
                                                diff_values
                                            ),
                                            95,
                                        )

                                        if (
                                            not np.isfinite(
                                                diff_abs
                                            )
                                            or diff_abs == 0
                                        ):

                                            diff_abs = 1.0

                                        norm = Normalize(
                                            vmin=-diff_abs,
                                            vmax=diff_abs,
                                        )

                                        cmap = cm.get_cmap(
                                            "RdBu_r"
                                        )

                                        rgba_diff = (
                                            cmap(
                                                norm(
                                                    np.clip(
                                                        diff,
                                                        -diff_abs,
                                                        diff_abs,
                                                    )
                                                )
                                            )
                                            * 255
                                        ).astype(
                                            np.uint8
                                        )

                                        rgba_diff[
                                            ~valid_diff,
                                            3,
                                        ] = 0

                                        diff_bounds = [
                                            [
                                                scen_src.bounds.bottom,
                                                scen_src.bounds.left,
                                            ],
                                            [
                                                scen_src.bounds.top,
                                                scen_src.bounds.right,
                                            ],
                                        ]

                                        diff_map = folium.Map(
                                            location=[
                                                (
                                                    diff_bounds[0][0]
                                                    + diff_bounds[1][0]
                                                )
                                                / 2,
                                                (
                                                    diff_bounds[0][1]
                                                    + diff_bounds[1][1]
                                                )
                                                / 2,
                                            ],
                                            zoom_start=9,
                                            control_scale=True,
                                        )

                                        Fullscreen().add_to(
                                            diff_map
                                        )

                                        folium.raster_layers.ImageOverlay(
                                            image=rgba_diff,
                                            bounds=diff_bounds,
                                            opacity=0.75,
                                            interactive=True,
                                            cross_origin=False,
                                        ).add_to(
                                            diff_map
                                        )

                                        st.caption(
                                            "Scenario: "
                                            f"{scenario_key.split('/')[-1]}  "
                                            " | Reference: "
                                            f"{reference_key.split('/')[-1]}"
                                        )

                                        st_folium(
                                            diff_map,
                                            width=None,
                                            height=550,
                                        )

            except Exception as exc:

                st.warning(
                    "Could not calculate the difference map:\n"
                    f"{exc}"
                )


# ============================================================
# SCENARIO BIOMASS / HARVEST TIME SERIES
# ============================================================

if (
    selected_source == "Scenario"
    and not active_geojson_df.empty
):

    st.subheader(
        "Mussel biomass and estimated harvest"
    )

    ts_df_tmp = (
        active_geojson_df.copy()
    )

    if ts_df_tmp.empty:

        st.info(
            "No biomass time-series data available."
        )

    else:

        # ----------------------------------------------------
        # Average biomass
        # ----------------------------------------------------

        avg_tmp = (
            ts_df_tmp.groupby(
                "time",
                as_index=False,
            )["value"]
            .mean()
            .rename(
                columns={
                    "value": "biomass"
                }
            )
        )

        # ----------------------------------------------------
        # Number of farms / points
        # ----------------------------------------------------

        n_farms = (
            ts_df_tmp[
                [
                    "lon",
                    "lat",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        avg_tmp["harvest"] = (
            avg_tmp["biomass"]
            * n_farms
        )

        # ----------------------------------------------------
        # Biomass chart
        # ----------------------------------------------------

        biomass_chart = (
            alt.Chart(
                avg_tmp
            )
            .mark_line()
            .encode(
                x=alt.X(
                    "time:T",
                    title="Time",
                ),
                y=alt.Y(
                    "biomass:Q",
                    title="Average biomass",
                    scale=alt.Scale(
                        type="log"
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "time:T",
                        title="Time",
                    ),
                    alt.Tooltip(
                        "biomass:Q",
                        title="Average biomass",
                        format=".3f",
                    ),
                ],
            )
            .properties(
                height=350,
            )
        )

        st.altair_chart(
            biomass_chart,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Harvest chart
        # ----------------------------------------------------

        harvest_chart = (
            alt.Chart(
                avg_tmp
            )
            .mark_bar()
            .encode(
                x=alt.X(
                    "time:T",
                    title="Time",
                ),
                y=alt.Y(
                    "harvest:Q",
                    title="Estimated harvest",
                ),
                tooltip=[
                    alt.Tooltip(
                        "time:T",
                        title="Time",
                    ),
                    alt.Tooltip(
                        "harvest:Q",
                        title="Estimated harvest",
                        format=".3f",
                    ),
                ],
            )
            .properties(
                height=350,
            )
        )

        st.altair_chart(
            harvest_chart,
            use_container_width=True,
        )


# ============================================================
# POINT SELECTION
# ============================================================

clicked = None

if map_data:

    clicked = (
        map_data.get(
            "last_object_clicked"
        )
        or map_data.get(
            "last_clicked"
        )
    )


if (
    clicked
    and not active_geojson_df.empty
):

    click_lat = clicked.get(
        "lat"
    )

    click_lon = clicked.get(
        "lng"
    )

    if (
        click_lat is not None
        and click_lon is not None
    ):

        points = (
            active_geojson_df[
                [
                    "lon",
                    "lat",
                ]
            ]
            .drop_duplicates()
            .copy()
        )

        points["distance"] = np.sqrt(
            (
                points["lon"]
                - click_lon
            ) ** 2
            +
            (
                points["lat"]
                - click_lat
            ) ** 2
        )

        nearest = points.loc[
            points["distance"].idxmin()
        ]

        selected_lon = nearest["lon"]
        selected_lat = nearest["lat"]

        point_ts = active_geojson_df[
            (
                active_geojson_df["lon"]
                == selected_lon
            )
            &
            (
                active_geojson_df["lat"]
                == selected_lat
            )
        ].sort_values(
            "time"
        )

        st.subheader(
            "Selected mussel-farm point"
        )

        st.write(
            f"Location: "
            f"{selected_lat:.5f}, "
            f"{selected_lon:.5f}"
        )

        point_chart = (
            alt.Chart(
                point_ts
            )
            .mark_line()
            .encode(
                x=alt.X(
                    "time:T",
                    title="Time",
                ),
                y=alt.Y(
                    "value:Q",
                    title="Biomass",
                ),
                tooltip=[
                    alt.Tooltip(
                        "time:T",
                        title="Time",
                    ),
                    alt.Tooltip(
                        "value:Q",
                        title="Biomass",
                        format=".3f",
                    ),
                ],
            )
            .properties(
                height=400,
            )
        )

        st.altair_chart(
            point_chart,
            use_container_width=True,
        )

        st.dataframe(
            point_ts,
            use_container_width=True,
        )


# ============================================================
# DATASET SUMMARY
# ============================================================

with st.expander(
    "Data availability"
):

    availability_rows = []

    for folder in [
        "ScenM0",
        "ScenM2",
        "ScenM3",
    ]:

        folder_df = _files_for_folder(
            folder
        )

        if folder_df.empty:
            continue

        for variable in (
            folder_df["variable"]
            .dropna()
            .unique()
        ):

            variable_df = folder_df[
                folder_df["variable"]
                == variable
            ]

            availability_rows.append(
                {
                    "Folder": folder,
                    "Variable": variable,
                    "Files": len(
                        variable_df
                    ),
                    "First time": variable_df[
                        "time"
                    ].min(),
                    "Last time": variable_df[
                        "time"
                    ].max(),
                }
            )

    if availability_rows:

        availability_df = pd.DataFrame(
            availability_rows
        ).sort_values(
            [
                "Folder",
                "Variable",
            ]
        )

        st.dataframe(
            availability_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No data availability information."
        )


# ============================================================
# DATA SOURCE INFORMATION
# ============================================================

with st.expander(
    "Public data source"
):

    st.write(
        f"**Public base URL:** `{PUBLIC_BASE_URL}`"
    )

    st.write(
        f"**Manifest:** `{PUBLIC_MANIFEST_URL}`"
    )

    st.write(
        f"**GeoTIFF URL:** `{PUBLIC_GEOTIFF_URL}`"
    )

    st.write(
        f"**GeoJSON URL:** `{PUBLIC_GEOJSON_URL}`"
    )

    st.write(
        f"**Reference folder:** `{REFERENCE_FOLDER}`"
    )

    st.write(
        "**Baseline-only variables:** "
        + ", ".join(
            sorted(
                BASELINE_ONLY_VARIABLES
            )
        )
    )

