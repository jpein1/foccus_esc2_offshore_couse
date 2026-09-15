# Offshore Wind–Aquaculture Digital Twin Dashboard
# by Johannes Pein, Helmholtz-Zentrum hereon
# johannes.pein@hereon.de
# Website: https://foccus-esc2-bluedream-lta.streamlit.app/

## Overview

This repository contains an interactive **Streamlit dashboard** for visualising and exploring results from an environmental digital twin developed for offshore multi-use planning. The application enables users to interactively compare environmental conditions and aquaculture production across multiple offshore development scenarios using spatial maps, time series and scenario comparison plots.

The dashboard was developed as part of the study:

> **Environmental Digital Twins for Scenario-Based Offshore Wind–Aquaculture Planning**

The underlying modelling framework combines:

* a coupled hydrodynamic–wave–biogeochemical model,
* a Dynamic Energy Budget (DEB) model for blue mussel (*Mytilus edulis*) growth,
* scenario-based offshore farm layouts,
* interactive geospatial visualisation using Streamlit.

The case study investigates blue mussel cultivation within the **Meerwind offshore wind farm** in the German Bight (North Sea) and demonstrates how environmental digital twins can support offshore planning, stakeholder communication and adaptive marine spatial planning.

---

# Scientific Background

The rapid expansion of offshore wind energy is increasing competition for marine space while simultaneously creating opportunities for integrated multi-use concepts. One promising approach is the co-location of offshore wind farms with low-trophic aquaculture such as blue mussel cultivation.

This dashboard serves as the visual interface to a digital twin framework that links:

* physical ocean conditions,
* ecosystem dynamics,
* mussel growth,
* aquaculture production,
* environmental impacts.

Users can compare alternative offshore farm configurations ("what-if" scenarios) and evaluate their influence on:

* mussel biomass production,
* estimated harvest,
* chlorophyll-a dynamics,
* dissolved oxygen,
* environmental differences relative to a reference simulation.

Rather than providing a single optimal solution, the dashboard is intended to support transparent comparison between alternative offshore planning scenarios.

---

# Repository Structure

```
project/

│
├── app.py
├── environment.yml
├── README.md
├── LICENSE
│
├── geojson/
│     Scenario_1.geojson
│     Scenario_2.geojson
│     ...
│
├── geotiff/
│     salt_geotiff_ScenM0/
│     temp_geotiff_ScenM0/
│     chla_geotiff_ScenM0/
│     oxy_geotiff_ScenM0/
│     chla_geotiff_ScenM2/
│     ...
│
└── data/
      (optional additional files)
```

---

# Installation

## 1. Clone the repository

```
git clone https://github.com/<repository>.git
cd <repository>
```

---

## 2. Create the Conda environment

The dashboard was developed using **Miniforge/Conda**.

```
conda env create -f environment.yml
```

Activate the environment

```
conda activate geo-app
```

---

## 3. Launch the dashboard

```
streamlit run app.py
```

A browser window should automatically open at

```
http://localhost:8501
```

---

# Data

The application requires two data types.

## GeoTIFF

Spatial environmental fields including

* salinity
* temperature
* chlorophyll-a
* dissolved oxygen

for baseline and scenario simulations.

---

## GeoJSON

Scenario-dependent mussel farm outputs containing

* farm locations
* harvest biomass
* complete time series

These data are used for the interactive aquaculture analyses.

---

# Full Dataset

The GitHub repository contains either

* a reduced demonstration dataset

or

* no model output.

The complete simulation dataset is available separately at

**DOI: (to be inserted after publication)**

After downloading, extract the folders so that the repository structure becomes

```
project/

geojson/
geotiff/
```

No further configuration is required.

---

# Using the Dashboard

The dashboard consists of three complementary components.

## 1. Environmental Maps

The upper panel displays GeoTIFF rasters representing environmental model output.

Available variables include

* Salinity
* Temperature
* Chlorophyll-a
* Dissolved oxygen

Users may switch between

* Baseline simulation
* Alternative offshore planning scenarios

using the control panel.

---

## 2. Aquaculture Analysis

Scenario-specific GeoJSON files provide

* mussel farm locations,
* average harvest biomass,
* production time series,
* estimated harvest.

Multiple scenarios can be displayed simultaneously to compare production trajectories.

---

## 3. Environmental Difference Maps

For biological variables the dashboard computes

```
Scenario − Reference
```

where

* the reference corresponds to the baseline biological simulation (ScenM0),
* the selected scenario is compared for the same simulation time.

Difference maps allow users to identify areas of increased or decreased chlorophyll-a or oxygen relative to the reference simulation.

---

# Typical Workflow

A recommended workflow is

1. Select an environmental variable.
2. Choose Baseline or Scenario mode.
3. Select a simulation time.
4. Display one or several aquaculture scenarios.
5. Compare harvest trajectories.
6. Inspect environmental differences relative to the baseline.
7. Explore individual farm time series by clicking map locations.

---

# Requirements

The application has been tested using

* Python 3.13
* Streamlit
* Rasterio
* Folium
* Streamlit-Folium
* GeoPandas
* Pandas
* NumPy
* Matplotlib
* Altair

The supplied `environment.yml` reproduces the development environment.

---

# Citation

If you use this software or the accompanying dataset, please cite

**(Paper citation to be inserted after publication)**

and

**(Zenodo dataset DOI to be inserted after publication)**

---

# License

This software is licensed under the Apache License 2.0.

See the LICENSE file for details.

# Contact

For questions regarding the modelling framework, dashboard or dataset, please contact the corresponding author.
