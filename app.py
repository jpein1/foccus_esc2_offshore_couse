import io
import json
import os
from datetime import datetime, date, time, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import altair as alt
import folium
import matplotlib 
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from streamlit_folium import st_folium


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="ESC2 Blue Economy",
    page_icon="🌊",
    layout="wide",
)


# =============================================================================
# PUBLIC EDITO DATA CONFIGURATION
# =============================================================================

# Public MinIO endpoint.
#
# Objects are available below:
#
# https://minio.dive.edito.eu/project-foccus/
#     Hereon/
#         ESC2_blue_economy/
#
PUBLIC_OBJECT_BASE_URL = os.getenv(
    "PUBLIC_OBJECT_BASE_URL",
    "https://minio.dive.edito.eu/project-foccus",
).rstrip("/")

PROJECT_PREFIX = os.getenv(
    "PROJECT_PREFIX",
    "Hereon/ESC2_blue_economy",
).strip("/")


# -------------------------------------------------------------------------
# Regular timestamp configuration
# -------------------------------------------------------------------------
#
# The filenames have the form:
#
#   variable_YYYYMMDDTHHMMSS.tif
#
# Example:
#
#   chla_20200501T000000.tif
#
# Change DATA_INTERVAL_HOURS if the model output is not hourly.
#
DATA_INTERVAL_HOURS = float(
    os.getenv("DATA_INTERVAL_HOURS", "1")
)

# Default first timestamp.
DATA_START = os.getenv(
    "DATA_START",
    "2020-05-01T00:00:00",
)

# Optional default last timestamp.  The UI also allows the user to
# select a date range.
DATA_END = os.getenv(
    "DATA_END",
    "2020-05-31T23:00:00",
)


# =============================================================================
# DATA SET STRUCTURE
# =============================================================================

# Variables that are reference/baseline variables.
#
# These MUST always come from ScenM0.
REFERENCE_ONLY_VARIABLES = {
    "salt",
    "temp",
}

# Variables that have scenario-specific outputs.
#
# Add other scenario variables here if they exist in the project.
SCENARIO_VARIABLES = {
    "chla",
    "oxy",
}

# Scenarios shown in the UI.
SCENARIOS = {
    "Baseline (ScenM0)": "ScenM0",
    "Scenario 1 (ScenM2)": "ScenM2",
    "Scenario 2 (ScenM3)": "ScenM3",
}


# =============================================================================
# PUBLIC OBJECT URL HELPERS
# =============================================================================

def public_project_url(relative_key: str) -> str:
    """
    Build the public HTTPS URL for an object in the EDITO project.

    Example:
        geotiff/ScenM0/chla_20200501T000000.tif

    becomes:
        https://minio.dive.edito.eu/project-foccus/
        Hereon/ESC2_blue_economy/
        geotiff/ScenM0/chla_20200501T000000.tif
    """
    relative_key = relative_key.lstrip("/")

    return (
        f"{PUBLIC_OBJECT_BASE_URL}/"
        f"{PROJECT_PREFIX}/"
        f"{quote(relative_key, safe='/')}"
    )


def geotiff_url(
    scenario: str,
    variable: str,
    timestamp: datetime,
) -> str:
    """
    Construct a GeoTIFF URL.

    IMPORTANT:
    - salt/temp are forced to ScenM0.
    - scenario variables use the requested scenario.
    """
    if variable in REFERENCE_ONLY_VARIABLES:
        scenario = "ScenM0"

    timestamp_string = timestamp.strftime("%Y%m%dT%H%M%S")

    filename = f"{variable}_{timestamp_string}.tif"

    return public_project_url(
        f"geotiff/{scenario}/{filename}"
    )


def csv_url(filename: str) -> str:
    return public_project_url(filename)


def geojson_url(filename: str) -> str:
    return public_project_url(f"geojson/{filename}")


# =============================================================================
# HTTP ACCESS
# =============================================================================

def _http_request(
    url: str,
    method: str = "GET",
    timeout: int = 60,
):
    """
    Make an anonymous HTTP request to the public EDITO object store.

    No S3 credentials are required.

    This deliberately does NOT use:
        boto3
        ListObjectsV2
        S3 bucket listing
    """
    request = Request(
        url,
        method=method,
        headers={
            "User-Agent": "ESC2-Blue-Economy-Streamlit-App/1.0",
        },
    )

    return urlopen(request, timeout=timeout)


@st.cache_data(ttl=3600, show_spinner=False)
def public_object_exists(url: str) -> bool:
    """
    Check whether a public object exists.

    HEAD is used so we do not download an entire GeoTIFF merely to check
    whether it exists.

    Some object stores may not support HEAD correctly. In that case we
    fall back to a small GET request.
    """
    try:
        with _http_request(url, method="HEAD", timeout=30):
            return True

    except HTTPError as exc:
        if exc.code in (404, 410):
            return False

        # Some servers reject HEAD. Try GET below.
        if exc.code not in (400, 403, 405):
            return False

    except (URLError, TimeoutError, OSError):
        return False

    # Fallback GET.
    try:
        with _http_request(url, method="GET", timeout=30) as response:
            response.read(1)
            return True

    except (HTTPError, URLError, TimeoutError, OSError):
        return False


@st.cache_data(ttl=3600, show_spinner=False)
def download_public_object(url: str) -> bytes:
    """
    Download a public object anonymously.
    """
    with _http_request(url, method="GET", timeout=180) as response:
        return response.read()


# =============================================================================
# GEO DATA LOADING
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def read_geotiff(url: str):
    """
    Read a public GeoTIFF into memory and return:
        array, metadata, bounds
    """
    data = download_public_object(url)

    with rasterio.MemoryFile(data) as memfile:
        with memfile.open() as dataset:
            array = dataset.read(1).astype(float)
            metadata = dataset.meta.copy()
            bounds = dataset.bounds
            transform = dataset.transform
            crs = dataset.crs
            nodata = dataset.nodata

    if nodata is not None:
        array[array == nodata] = np.nan

    return (
        array,
        metadata,
        bounds,
        transform,
        crs,
    )


# =============================================================================
# CSV / GEOJSON LOADING
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def read_public_csv(url: str) -> pd.DataFrame:
    data = download_public_object(url)
    return pd.read_csv(io.BytesIO(data))


@st.cache_data(ttl=3600, show_spinner=False)
def read_public_json(url: str):
    data = download_public_object(url)
    return json.loads(data.decode("utf-8"))


# =============================================================================
# TIMESTAMP HANDLING
# =============================================================================

def parse_timestamp(value: str) -> datetime:
    """
    Parse ISO timestamp strings such as:
        2020-05-01T00:00:00
    """
    value = value.strip()

    if value.endswith("Z"):
        value = value[:-1]

    return datetime.fromisoformat(value)


def generate_timestamps(
    start: datetime,
    end: datetime,
):
    """
    Generate the regular model-output timestamps.

    DATA_INTERVAL_HOURS is configurable through the environment.
    """
    interval_seconds = int(DATA_INTERVAL_HOURS * 3600)

    if interval_seconds <= 0:
        raise ValueError(
            "DATA_INTERVAL_HOURS must be greater than zero."
        )

    step = timedelta(seconds=interval_seconds)

    timestamps = []

    current = start

    while current <= end:
        timestamps.append(current)
        current += step

    return timestamps


def timestamp_label(timestamp: datetime) -> str:
    return timestamp.strftime("%Y-%m-%d %H:%M")


# =============================================================================
# AVAILABLE TIMESTAMPS
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def find_available_timestamps(
    variable: str,
    scenario: str,
    start_string: str,
    end_string: str,
):
    """
    Probe the deterministic public URLs for the requested variable.

    There is deliberately no bucket listing here.

    For salt/temp, scenario is automatically forced to ScenM0.
    """
    start = parse_timestamp(start_string)
    end = parse_timestamp(end_string)

    candidates = generate_timestamps(start, end)

    available = []

    for timestamp in candidates:
        url = geotiff_url(
            scenario=scenario,
            variable=variable,
            timestamp=timestamp,
        )

        if public_object_exists(url):
            available.append(timestamp)

    return available


# =============================================================================
# DIFFERENCE CALCULATION
# =============================================================================

def calculate_difference(
    scenario_array: np.ndarray,
    reference_array: np.ndarray,
) -> np.ndarray:
    """
    Calculate scenario minus baseline.

    The two rasters are expected to use the same grid.
    """
    if scenario_array.shape != reference_array.shape:
        raise ValueError(
            "Scenario and reference rasters have different dimensions: "
            f"{scenario_array.shape} vs {reference_array.shape}"
        )

    return scenario_array - reference_array


# =============================================================================
# MAP / RASTER HELPERS
# =============================================================================

def raster_to_dataframe(
    array: np.ndarray,
    transform,
    bounds,
    max_points: int = 5000,
):
    """
    Convert a raster to a dataframe suitable for plotting.

    Downsampling is applied for large rasters.
    """
    rows, cols = array.shape

    valid = np.isfinite(array)

    if not valid.any():
        return pd.DataFrame(
            columns=["lon", "lat", "value"]
        )

    row_indices, col_indices = np.where(valid)

    values = array[row_indices, col_indices]

    if len(values) > max_points:
        step = max(1, len(values) // max_points)

        row_indices = row_indices[::step]
        col_indices = col_indices[::step]
        values = values[::step]

    xs, ys = rasterio.transform.xy(
        transform,
        row_indices,
        col_indices,
        offset="center",
    )

    return pd.DataFrame(
        {
            "lon": np.asarray(xs),
            "lat": np.asarray(ys),
            "value": values,
        }
    )


def raster_center(bounds):
    return (
        (bounds.bottom + bounds.top) / 2,
        (bounds.left + bounds.right) / 2,
    )


def make_folium_map(
    array: np.ndarray,
    bounds,
    transform,
    title: str,
):
    """
    Create a simple Folium map with the raster represented as points.

    The point representation avoids writing temporary raster files to
    the container.
    """
    center_lat, center_lon = raster_center(bounds)

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="CartoDB positron",
        control_scale=True,
    )

    plot_df = raster_to_dataframe(
        array,
        transform,
        bounds,
    )

    if plot_df.empty:
        return fmap

    values = plot_df["value"].to_numpy()

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    if np.isclose(vmin, vmax):
        normalized = np.full_like(
            values,
            0.5,
            dtype=float,
        )
    else:
        normalized = (
            (values - vmin) /
            (vmax - vmin)
        )

    cmap = matplotlib.colormaps["viridis"]

    for lon, lat, value, norm_value in zip(
        plot_df["lon"],
        plot_df["lat"],
        plot_df["value"],
        normalized,
    ):
        rgba = cmap(float(norm_value))

        color = (
            f"#{int(rgba[0] * 255):02x}"
            f"{int(rgba[1] * 255):02x}"
            f"{int(rgba[2] * 255):02x}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=(
                f"{title}<br>"
                f"Value: {value:.4g}"
            ),
        ).add_to(fmap)

    return fmap


# =============================================================================
# VARIABLE LABELS / UNITS
# =============================================================================

VARIABLE_INFO = {
    "salt": {
        "label": "Salinity",
        "unit": "",
    },
    "temp": {
        "label": "Temperature",
        "unit": "°C",
    },
    "chla": {
        "label": "Chlorophyll-a",
        "unit": "",
    },
    "oxy": {
        "label": "Oxygen",
        "unit": "",
    },
}


def variable_label(variable: str) -> str:
    return VARIABLE_INFO.get(
        variable,
        {"label": variable, "unit": ""},
    )["label"]


def variable_unit(variable: str) -> str:
    return VARIABLE_INFO.get(
        variable,
        {"label": variable, "unit": ""},
    )["unit"]


# =============================================================================
# HARVEST GEOJSON
# =============================================================================

HARVEST_GEOJSON = {
    "ScenM2": "harvest_timeseries_scenario_Scen_M2.geojson",
    "ScenM3": "harvest_timeseries_scenario_Scen_M3.geojson",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_harvest_geojson(scenario: str):
    filename = HARVEST_GEOJSON.get(scenario)

    if filename is None:
        return None

    url = geojson_url(filename)

    if not public_object_exists(url):
        return None

    try:
        return read_public_json(url)
    except Exception:
        return None


# =============================================================================
# MEERWIND MONOPILE CSV
# =============================================================================

MEERWIND_CSV = "Meerwind_monopiles_lonlat.csv"


@st.cache_data(ttl=3600, show_spinner=False)
def load_meerwind_locations():
    url = csv_url(MEERWIND_CSV)

    if not public_object_exists(url):
        return None

    try:
        return read_public_csv(url)
    except Exception:
        return None


def add_monopile_locations(fmap, dataframe):
    """
    Add Meerwind monopile locations where latitude/longitude columns
    can be identified.
    """
    if dataframe is None or dataframe.empty:
        return

    columns_lower = {
        str(column).lower(): column
        for column in dataframe.columns
    }

    lat_column = None
    lon_column = None

    for candidate in (
        "lat",
        "latitude",
        "y",
    ):
        if candidate in columns_lower:
            lat_column = columns_lower[candidate]
            break

    for candidate in (
        "lon",
        "longitude",
        "x",
    ):
        if candidate in columns_lower:
            lon_column = columns_lower[candidate]
            break

    if lat_column is None or lon_column is None:
        return

    for _, row in dataframe.iterrows():
        try:
            lat = float(row[lat_column])
            lon = float(row[lon_column])
        except (ValueError, TypeError):
            continue

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="black",
            fill=True,
            fill_color="white",
            fill_opacity=1,
            weight=1,
            popup="Meerwind monopile",
        ).add_to(fmap)


# =============================================================================
# GEOJSON MAP OVERLAY
# =============================================================================

def add_geojson_overlay(fmap, geojson_data):
    if not geojson_data:
        return

    try:
        folium.GeoJson(
            geojson_data,
            name="Harvest time series",
            tooltip=folium.GeoJsonTooltip(
                fields=[],
                aliases=[],
                localize=True,
            ),
        ).add_to(fmap)
    except Exception:
        # Do not let an optional overlay prevent the raster map from loading.
        pass


# =============================================================================
# SESSION STATE
# =============================================================================

if "selected_timestamp" not in st.session_state:
    st.session_state.selected_timestamp = None


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("ESC2 Blue Economy")

st.sidebar.markdown(
    """
This application reads the public EDITO/MinIO objects directly over HTTPS.

No S3 bucket listing or credentials are required.
"""
)

st.sidebar.markdown("---")


# Scenario selector.
scenario_label = st.sidebar.selectbox(
    "Scenario",
    list(SCENARIOS.keys()),
)

selected_scenario = SCENARIOS[scenario_label]


# Variable selector.
available_variables = sorted(
    REFERENCE_ONLY_VARIABLES | SCENARIO_VARIABLES
)

selected_variable = st.sidebar.selectbox(
    "Variable",
    available_variables,
    format_func=variable_label,
)


# =============================================================================
# DATE RANGE
# =============================================================================

default_start = parse_timestamp(DATA_START)
default_end = parse_timestamp(DATA_END)

st.sidebar.markdown("### Time range")

selected_start_date = st.sidebar.date_input(
    "Start date",
    value=default_start.date(),
    min_value=date(2020, 1, 1),
)

selected_end_date = st.sidebar.date_input(
    "End date",
    value=default_end.date(),
    min_value=date(2020, 1, 1),
)

if selected_end_date < selected_start_date:
    st.sidebar.error(
        "The end date must be on or after the start date."
    )
    st.stop()


# Preserve the configured time-of-day at the beginning/end.
query_start = datetime.combine(
    selected_start_date,
    time(
        default_start.hour,
        default_start.minute,
        default_start.second,
    ),
)

query_end = datetime.combine(
    selected_end_date,
    time(
        default_end.hour,
        default_end.minute,
        default_end.second,
    ),
)


# =============================================================================
# DIFFERENCE MODE
# =============================================================================

difference_requested = False

if selected_variable not in REFERENCE_ONLY_VARIABLES:
    difference_requested = st.sidebar.checkbox(
        "Show difference from ScenM0",
        value=False,
        help=(
            "For scenario variables, calculate "
            "scenario minus ScenM0."
        ),
    )

else:
    st.sidebar.info(
        "Salt and temperature are baseline/reference variables "
        "and are always read from ScenM0."
    )


# =============================================================================
# INFORMATION
# =============================================================================

with st.sidebar.expander("Data source"):
    st.write(
        "Project:",
        f"{PUBLIC_OBJECT_BASE_URL}/{PROJECT_PREFIX}",
    )

    st.write(
        "GeoTIFF directory:",
        f"{PUBLIC_OBJECT_BASE_URL}/{PROJECT_PREFIX}/geotiff",
    )

    st.write(
        "Timestamp interval:",
        f"{DATA_INTERVAL_HOURS:g} hour(s)",
    )

    if selected_variable in REFERENCE_ONLY_VARIABLES:
        st.write(
            "Data source:",
            "ScenM0",
        )
    else:
        st.write(
            "Data source:",
            selected_scenario,
        )


# =============================================================================
# HEADER
# =============================================================================

st.title("ESC2 Blue Economy")

st.markdown(
    """
Explore the ESC2 marine environmental model outputs from the
**Hereon / ESC2_blue_economy** project.
"""
)


# =============================================================================
# FIND AVAILABLE TIMES
# =============================================================================

start_string = query_start.isoformat()
end_string = query_end.isoformat()

with st.spinner(
    f"Checking public {variable_label(selected_variable)} files..."
):
    available_timestamps = find_available_timestamps(
        variable=selected_variable,
        scenario=selected_scenario,
        start_string=start_string,
        end_string=end_string,
    )


if not available_timestamps:
    st.error(
        "No GeoTIFF files were found for the selected variable, "
        "scenario and date range."
    )

    st.info(
        "The app constructs filenames according to the pattern "
        "`variable_YYYYMMDDTHHMMSS.tif`. Check DATA_INTERVAL_HOURS "
        "if the model output interval is different from the configured "
        "value."
    )

    st.stop()


# =============================================================================
# TIMESTAMP SELECTOR
# =============================================================================

timestamp_options = {
    timestamp_label(ts): ts
    for ts in available_timestamps
}

default_timestamp = (
    st.session_state.selected_timestamp
    if st.session_state.selected_timestamp in available_timestamps
    else available_timestamps[0]
)

selected_timestamp_label = st.selectbox(
    "Model timestamp",
    list(timestamp_options.keys()),
    index=list(timestamp_options.values()).index(
        default_timestamp
    ),
)

selected_timestamp = timestamp_options[
    selected_timestamp_label
]

st.session_state.selected_timestamp = selected_timestamp


# =============================================================================
# LOAD SELECTED RASTER
# =============================================================================

selected_url = geotiff_url(
    scenario=selected_scenario,
    variable=selected_variable,
    timestamp=selected_timestamp,
)

try:
    with st.spinner("Loading GeoTIFF..."):
        (
            selected_array,
            selected_metadata,
            selected_bounds,
            selected_transform,
            selected_crs,
        ) = read_geotiff(selected_url)

except Exception as exc:
    st.error(
        "Could not load the selected GeoTIFF."
    )

    st.code(
        selected_url,
        language="text",
    )

    st.exception(exc)

    st.stop()


# =============================================================================
# DIFFERENCE MAP
# =============================================================================

display_array = selected_array
display_title = (
    f"{variable_label(selected_variable)} — "
    f"{scenario_label}"
)

reference_url = None

if (
    difference_requested
    and selected_variable not in REFERENCE_ONLY_VARIABLES
    and selected_scenario != "ScenM0"
):
    reference_url = geotiff_url(
        scenario="ScenM0",
        variable=selected_variable,
        timestamp=selected_timestamp,
    )

    if not public_object_exists(reference_url):
        st.warning(
            "The corresponding ScenM0 reference GeoTIFF "
            "could not be found for this timestamp. "
            "Showing the scenario value instead."
        )

    else:
        try:
            with st.spinner("Loading ScenM0 reference..."):
                (
                    reference_array,
                    reference_metadata,
                    reference_bounds,
                    reference_transform,
                    reference_crs,
                ) = read_geotiff(reference_url)

            # Make sure the two rasters are on the same grid.
            same_shape = (
                selected_array.shape ==
                reference_array.shape
            )

            same_transform = (
                selected_transform ==
                reference_transform
            )

            same_crs = (
                selected_crs ==
                reference_crs
            )

            if not (same_shape and same_transform and same_crs):
                st.warning(
                    "The scenario and ScenM0 rasters do not have "
                    "identical grids. Difference calculation was skipped."
                )
            else:
                display_array = calculate_difference(
                    selected_array,
                    reference_array,
                )

                display_title = (
                    f"{variable_label(selected_variable)} — "
                    f"{scenario_label} − ScenM0"
                )

        except Exception as exc:
            st.warning(
                "The ScenM0 reference could not be loaded. "
                "Showing the selected scenario value instead."
            )

            with st.expander("Reference loading details"):
                st.exception(exc)


# =============================================================================
# SUMMARY METRICS
# =============================================================================

valid_values = display_array[
    np.isfinite(display_array)
]

st.markdown("### Selected output")

metric_columns = st.columns(4)

with metric_columns[0]:
    st.metric(
        "Scenario",
        scenario_label,
    )

with metric_columns[1]:
    st.metric(
        "Variable",
        variable_label(selected_variable),
    )

with metric_columns[2]:
    st.metric(
        "Timestamp",
        selected_timestamp.strftime(
            "%Y-%m-%d %H:%M"
        ),
    )

with metric_columns[3]:
    if valid_values.size:
        st.metric(
            "Valid cells",
            f"{valid_values.size:,}",
        )
    else:
        st.metric(
            "Valid cells",
            "0",
        )


# =============================================================================
# MAP
# =============================================================================

st.markdown("### Spatial distribution")

fmap = make_folium_map(
    array=display_array,
    bounds=selected_bounds,
    transform=selected_transform,
    title=display_title,
)

# Add optional Meerwind monopile locations.
monopile_data = load_meerwind_locations()

if monopile_data is not None:
    add_monopile_locations(
        fmap,
        monopile_data,
    )


# Add optional harvest GeoJSON for scenario maps.
if selected_scenario in ("ScenM2", "ScenM3"):
    harvest_geojson = load_harvest_geojson(
        selected_scenario
    )

    if harvest_geojson is not None:
        add_geojson_overlay(
            fmap,
            harvest_geojson,
        )


st_folium(
    fmap,
    width=None,
    height=650,
    returned_objects=[],
)


# =============================================================================
# STATISTICS
# =============================================================================

st.markdown("### Raster statistics")

if valid_values.size:
    stats_columns = st.columns(5)

    with stats_columns[0]:
        st.metric(
            "Minimum",
            f"{np.nanmin(valid_values):.4g}",
        )

    with stats_columns[1]:
        st.metric(
            "Maximum",
            f"{np.nanmax(valid_values):.4g}",
        )

    with stats_columns[2]:
        st.metric(
            "Mean",
            f"{np.nanmean(valid_values):.4g}",
        )

    with stats_columns[3]:
        st.metric(
            "Median",
            f"{np.nanmedian(valid_values):.4g}",
        )

    with stats_columns[4]:
        st.metric(
            "Std. dev.",
            f"{np.nanstd(valid_values):.4g}",
        )

else:
    st.warning(
        "The selected raster contains no valid data cells."
    )


# =============================================================================
# DATA URL
# =============================================================================

with st.expander("Public GeoTIFF URL"):
    st.code(
        selected_url,
        language="text",
    )

    if reference_url is not None:
        st.markdown("**ScenM0 reference:**")

        st.code(
            reference_url,
            language="text",
        )


# =============================================================================
# TIME SERIES OVERVIEW
# =============================================================================

st.markdown("### Available timestamps")

timestamp_df = pd.DataFrame(
    {
        "Timestamp": [
            ts.strftime("%Y-%m-%d %H:%M:%S")
            for ts in available_timestamps
        ],
        "Variable": [
            selected_variable
            for _ in available_timestamps
        ],
        "Scenario": [
            (
                "ScenM0"
                if selected_variable in REFERENCE_ONLY_VARIABLES
                else selected_scenario
            )
            for _ in available_timestamps
        ],
    }
)

st.dataframe(
    timestamp_df,
    use_container_width=True,
    hide_index=True,
)


# =============================================================================
# CONFIGURATION NOTE
# =============================================================================

DATA_INTERVAL_HOURS = float(
    os.getenv("DATA_INTERVAL_HOURS", "1")
)
