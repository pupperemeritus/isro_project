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
