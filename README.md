# S4 Data Visualization Tool

This tool allows you to visualize and analyze S4 (scintillation) data from GPS satellites.

## Features

1. Interactive Map: View S4 data on a map with customizable markers and heatmaps.
2. Time Series Plot: Analyze S4 values over time for specific satellites.
3. Skyplot: Visualize satellite positions and their corresponding S4 values.
4. Data Filtering: Filter data by time range, satellite ID, elevation, azimuth, and S4 threshold.

## Installation

1. Clone this repository
2. Install the required packages using poetry

# Usage

1. Run the Streamlit app:
2. Upload your CSV file containing S4 data.
3. Use the sidebar to select visualization type and apply filters.
4. Interact with the visualizations to explore your data.

# Data Format

The input CSV file should contain the following columns:

- GPS WN: GPS Week Number
- GPS TOW: GPS Time of Week
- SVID: Satellite Vehicle ID
- Elevation (degrees)
- Azimuth (degrees)
- S4: Scintillation Index

# Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# License

This project is licensed under the MIT License.

```
isro_project
├─ README.md
├─ __init__.py
├─ app
│  ├─ __init__.py
│  ├─ custom_css.py
│  ├─ logging_conf.py
│  ├─ main.py
│  ├─ map_creator.py
│  ├─ utils.py
│  └─ visualizations.py
├─ backend
├─ lstm.py
├─ model
│  ├─ __init__.py
│  └─ peephole_lstm.py
├─ poetry.lock
├─ prophet_model.py
├─ pyproject.toml
├─ reference_material
│  ├─ 01c3b453-f19c-4cbd-b6b0-d3cef50ce65b.pdf
│  ├─ 1320067c-49d0-42a4-80d5-d194c25ad166.pdf
│  ├─ 20101405.pdf
│  ├─ 2013_Spogli_JASTP_Clim_Braz.pdf
│  ├─ 20210325-ASR+-+0917-revision-R2.pdf
│  ├─ 24f0de0d-df4f-42ae-a92c-cbde5768f39d.pdf
│  ├─ 402d5554-e7c1-49cf-8ced-394e2f91b5c4.pdf
│  ├─ 4d92d1fb-705d-4031-8d3d-f4ab0ae4ab4b.pdf
│  ├─ 69333.pdf
│  ├─ Algorithms for Hyper-Parameter Optimization.pdf
│  ├─ An Autoregressive Integrated Moving Average (ARIMA) Based Forecasting of Ionospheric Total Electron Content at a low latitude Indian Location.pdf
│  ├─ An analysis of global ionospheric disturbances and scintillations during the strong magnetic storm in September 2017.pdf
│  ├─ ETASR_5863.pdf
│  ├─ Forecasting low‐latitude radio scintillation with 3‐D ionospheric plume models_ 1. Plume model.pdf
│  ├─ Forecasting low‐latitude radio scintillation with 3‐D ionospheric plume models_ 2. Scintillation calculation.pdf
│  ├─ Google_Vizier_A_Service_for_Black-Box_Optimization.pdf
│  ├─ High-resolution-global-weather-model-afno.pdf
│  ├─ Ionospheric Scintillations_ Indices and Modeling.pdf
│  ├─ Ionospheric Total Electron Content Forecasting at a Low-Latitude Indian Location Using a Bi-Long Short-Term Memory Deep Learning Approach.pdf
│  ├─ Modeling and scientific application of scintillation results.pdf
│  ├─ Modeling of ionospheric scintillation.pdf
│  ├─ NeurIPS-2022-earthformer-exploring-space-time-transformers-for-earth-system-forecasting-Paper-Conference.pdf
│  ├─ Nowcasting of Amplitude Ionospheric Scintillation Based on Machine Learning Techniques.pdf
│  ├─ On estimating the phase scintillation index using TEC provided by ISM and IGS professional GNSS receivers and machine learning.pdf
│  ├─ On the Relationship Between the Rate of Change of Total Electron Content Index (ROTI), Irregularity Strength (CkL), and the Scintillation Index (S4).pdf
│  ├─ Radio Science - 2022 - Li - Ionospheric Scintillation Monitoring With ROTI From Geodetic Receiver  Limitations and.pdf
│  ├─ The_Short-Term_Prediction_of_Low-Latitude_Ionospheric_Irregularities_Leveraging_a_Hybrid_Ensemble_Model.pdf
│  ├─ Unified Training of Universal Time Series Forecasting Transformers.pdf
│  ├─ aarons1985.pdf
│  ├─ afb154b9-3ae0-461c-9049-f6818f52539e.pdf
│  ├─ akiba2019.pdf
│  ├─ areviewofionosphericscintillationmodelsspriyadarshi.pdf
│  ├─ b76898ea-8acf-4bb0-bb63-3faa485b9b7b.pdf
│  ├─ basu1976.pdf
│  ├─ basu2002.pdf
│  ├─ bramley1967.pdf
│  ├─ briggs1963.pdf
│  ├─ c8d6d3e3-b202-44ed-8f36-3fb6dca96f2c.pdf
│  ├─ camps2017.pdf
│  ├─ compactgnss.pdf
│  ├─ da9c8611-5859-463d-8a17-f7eb59cebd93.pdf
│  ├─ depaula2003.pdf
│  ├─ franke1985.pdf
│  ├─ futureinternet-15-00255-v2.pdf
│  ├─ hinson1986.pdf
│  ├─ hir-22-351.pdf
│  ├─ humphreys2009.pdf
│  ├─ prophet.pdf
│  ├─ remotesensing-13-03732-v2.pdf
│  ├─ rumsey1975.pdf
│  ├─ s40645-017-0153-6.pdf
│  ├─ science.adi2336.pdf
│  ├─ sensors-20-02877-v3.pdf
│  └─ uscinski1985.pdf
├─ results.txt
├─ shared_scripts
├─ tests
│  └─ test_main.py
└─ todo.md

```