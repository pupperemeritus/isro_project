import enum
from datetime import date, time
from typing import List, Optional

from fastapi import Body, FastAPI, Path, Query
from pydantic import BaseModel, Field

app = FastAPI(
    title="S4 Data Plotter API",
    description="API for retrieving and visualizing S4 data for the Dynamic S4 Data Plotter application.",
    version="1.0.0",
)


# Enums
class VisualizationType(str, enum.Enum):
    Map = "Map"
    TimeSeries = "TimeSeries"
    Skyplot = "Skyplot"


class MapType(str, enum.Enum):
    ScatterHeatmap = "Scatter/Heatmap"
    TEC = "TEC"


class MapStyle(str, enum.Enum):
    OpenStreetMap = "open-street-map"
    CartoDarkMatter = "carto-darkmatter"
    CartoPositron = "carto-positron"
    StamenTerrain = "stamen-terrain"
    StamenToner = "stamen-toner"
    StamenWatercolor = "stamen-watercolor"


class DataPreview(BaseModel):
    # Define the structure of your data preview
    ...


class FilteredDataParams(BaseModel):
    file_name: str
    selected_date: date
    selected_time: time
    window: int = Field(10, ge=1, le=30)
    latitude_range: str
    longitude_range: str
    s4_threshold: float = Field(0.0, ge=0.0, le=1.0)
    svid: Optional[List[int]] = None


class FilteredData(BaseModel):
    data: List[dict]
    svid_options: List[int]


class PlotlyVisualization(BaseModel):
    data: List[dict]
    layout: dict


class MatplotlibVisualization(BaseModel):
    image: str


class Visualization(BaseModel):
    type: str
    data: PlotlyVisualization | MatplotlibVisualization


# API routes
@app.get("/files", response_model=List[str])
async def get_files() -> List[str]:
    """
    Retrieve available data files.
    Returns a list of all available S4 data files for analysis.
    """
    ...


@app.get("/data_preview/{file_name}", response_model=DataPreview)
async def get_data_preview(
    file_name: str = Path(..., description="Name of the file to preview")
) -> DataPreview:
    """
    Get data preview.
    Returns a preview of the selected data file.
    """
    ...


@app.get("/filtered_data", response_model=FilteredData)
async def get_filtered_data(
    file_name: str = Query(..., description="Name of the file to analyze"),
    selected_date: date = Query(
        ..., description="Selected date for filtering (YYYY-MM-DD)"
    ),
    selected_time: time = Query(
        ..., description="Selected time for filtering (HH:MM:SS)"
    ),
    window: int = Query(10, ge=1, le=30, description="Time window in minutes"),
    latitude_range: str = Query(
        ...,
        description="Latitude range (min,max)",
        regex=r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?$",
    ),
    longitude_range: str = Query(
        ...,
        description="Longitude range (min,max)",
        regex=r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?$",
    ),
    s4_threshold: float = Query(0.0, ge=0.0, le=1.0, description="S4 threshold value"),
    svid: Optional[List[int]] = Query(
        None, description="List of SVID values to include"
    ),
) -> FilteredData:
    """
    Get filtered S4 data.
    Returns filtered S4 data based on provided parameters.
    """
    ...


@app.get("/visualization/{viz_type}", response_model=Visualization)
async def get_visualization(
    viz_type: VisualizationType = Path(..., description="Type of visualization"),
    file_name: str = Query(..., description="Name of the file to visualize"),
    filtered_data_params: FilteredDataParams = Body(
        ..., description="JSON object containing all filtered data parameters"
    ),
    map_type: Optional[MapType] = Query(
        None, description="Type of map visualization (required for Map viz_type)"
    ),
    map_style: Optional[MapStyle] = Query(
        None, description="Style of the map (required for Map viz_type)"
    ),
    marker_size: Optional[int] = Query(
        None, description="Size of markers for Scatter/Heatmap"
    ),
    heatmap_size: Optional[int] = Query(
        None, description="Size of heatmap for Scatter/Heatmap"
    ),
    color_scale: Optional[str] = Query(
        None, description="Color scale for the visualization"
    ),
    bin_heatmap: Optional[bool] = Query(None, description="Whether to bin the heatmap"),
    selected_svid: Optional[int] = Query(
        None, description="Selected SVID for Time Series visualization"
    ),
) -> Visualization:
    """
    Get visualization data.
    Returns data for the requested visualization type.
    """
    ...
