# model/graphcast_dataloader.py
import logging
from typing import Any, Dict, Tuple, Iterator

import numpy as np
from darts.logging import get_logger
from graphcast import typed_graph, icosahedral_mesh
import polars as pl
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d

logger = get_logger(__name__)


class IonosphereGraphCastDataset:
    def __init__(
        self,
        dataframe: pl.DataFrame,
        sequence_length: int,
        prediction_horizon: int,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        grid_resolution: float,
        time_window: str,
        stride: int,
    ):
        self.df = dataframe.sort("IST_Time")
        self.time_window = time_window
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.grid_lat, self.grid_lon = create_grid(
            lat_range, lon_range, grid_resolution
        )
        self.num_grid_nodes = len(self.grid_lat) * len(self.grid_lon)
        self.mesh = icosahedral_mesh.get_last_triangular_mesh_for_sphere(splits=1)
        self._num_mesh_nodes = self.mesh.vertices.shape[0]
        self._grid_nodes_lat = self.grid_lat.reshape([-1]).astype(np.float32)
        self._grid_nodes_lon = self.grid_lon.reshape([-1]).astype(np.float32)

        self.grouped_data = list(
            self.df.group_by_dynamic("IST_Time", every=time_window)
        )
        self.sequences = [
            (i, i + sequence_length + prediction_horizon)
            for i in range(
                0,
                len(self.grouped_data) - sequence_length - prediction_horizon + 1,
                self.stride,
            )
        ]
        self.grid_lat_steps = len(self.grid_lat)
        self.grid_lon_steps = len(self.grid_lon)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start, end = self.sequences[idx]
        sequence_data = self.grouped_data[start : start + self.sequence_length]
        target_data = self.grouped_data[start + self.sequence_length : end]

        if not sequence_data or not target_data:
            raise ValueError("Sequence or target data is empty for index %d" % idx)

        sequence_s4_list = [
            interpolate_to_grid(
                group[1][["Latitude", "Longitude"]].to_numpy(),
                group[1]["Vertical S4"].to_numpy(),
                self.grid_lat,
                self.grid_lon,
            )
            for group in sequence_data
        ]
        sequence_phase_list = [
            interpolate_to_grid(
                group[1][["Latitude", "Longitude"]].to_numpy(),
                group[1]["Vertical Scintillation Phase"].to_numpy(),
                self.grid_lat,
                self.grid_lon,
            )
            for group in sequence_data
        ]

        target_s4_list = [
            interpolate_to_grid(
                group[1][["Latitude", "Longitude"]].to_numpy(),
                group[1]["Vertical S4"].to_numpy(),
                self.grid_lat,
                self.grid_lon,
            )
            for group in target_data
        ]
        target_phase_list = [
            interpolate_to_grid(
                group[1][["Latitude", "Longitude"]].to_numpy(),
                group[1]["Vertical Scintillation Phase"].to_numpy(),
                self.grid_lat,
                self.grid_lon,
            )
            for group in target_data
        ]

        sequence_s4 = np.stack(sequence_s4_list).astype(np.float32)
        sequence_phase = np.stack(sequence_phase_list).astype(np.float32)
        target_s4 = np.stack(target_s4_list).astype(np.float32)
        target_phase = np.stack(target_phase_list).astype(np.float32)

        grid_node_features = np.stack([sequence_s4, sequence_phase], axis=-1)
        graph = self._create_graph(grid_node_features[0])
        graph_input = graph._replace(
            nodes={
                "grid_nodes": graph.nodes["grid_nodes"]._replace(
                    features=np.transpose(grid_node_features, [1, 0, 2])
                )
            }
        )
        target_tensor = np.stack([target_s4, target_phase], axis=-1)
        return graph_input, target_tensor.astype(np.float32)

    def _create_graph(self, grid_node_features_single_timestep):
        grid_node_set = typed_graph.NodeSet(
            n_node=np.array([self.num_grid_nodes]),
            features=grid_node_features_single_timestep,
        )
        mesh_node_set = typed_graph.NodeSet(
            n_node=np.array([self._num_mesh_nodes]),
            features=np.zeros((self._num_mesh_nodes, 1), dtype=np.float32),
        )
        graph = typed_graph.TypedGraph(
            context=typed_graph.Context(n_graph=np.array([1]), features=()),
            nodes={"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set},
            edges={},
        )
        return graph


def create_grid(lat_range, lon_range, grid_resolution):
    lat_start, lat_end = lat_range
    lon_start, lon_end = lon_range
    grid_lat = np.arange(lat_start, lat_end + grid_resolution, grid_resolution)
    grid_lon = np.arange(lon_start, lon_end + grid_resolution, grid_resolution)
    return grid_lat, grid_lon


def interpolate_to_grid(points, values, grid_lat, grid_lon, min_points=4):
    if len(points) < min_points:
        return direct_binning(points, values, grid_lat, grid_lon)
    try:
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
        interpolated = griddata(
            points, values, (grid_x, grid_y), method="linear", fill_value=np.nan
        )
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            binned = direct_binning(points, values, grid_lat, grid_lon)
            interpolated[nan_mask] = binned[nan_mask]
        return interpolated.flatten()
    except Exception as e:
        return direct_binning(points, values, grid_lat, grid_lon)


def direct_binning(points, values, grid_lat, grid_lon):
    lat_min, lat_max = np.min(grid_lat), np.max(grid_lat)
    lon_min, lon_max = np.min(grid_lon), np.max(grid_lon)
    binned, _, _, _ = binned_statistic_2d(
        points[:, 1],
        points[:, 0],
        values,
        bins=[len(grid_lon), len(grid_lat)],
        statistic="mean",
        range=[[lon_min, lon_max], [lat_min, lat_max]],
    )
    binned = np.nan_to_num(binned.T, nan=0)
    return binned.flatten()


class IonosphereGraphCastDataModule:
    def __init__(
        self,
        dataframe: pl.DataFrame,
        sequence_length: int,
        prediction_horizon: int,
        grid_lat_range: Tuple[float, float],
        grid_lon_range: Tuple[float, float],
        grid_resolution: float,
        time_window: str,
        stride: int,
        batch_size: int,
        num_workers: int,
    ):
        self.hparams = {
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,
            "grid_lat_range": grid_lat_range,
            "grid_lon_range": grid_lon_range,
            "grid_resolution": grid_resolution,
            "time_window": time_window,
            "stride": stride,
            "batch_size": batch_size,
            "num_workers": num_workers,
        }
        self.dataframe = dataframe

    def setup(self):
        train_df, val_df, test_df = self._split_data()
        dataset_params = dict(
            sequence_length=self.hparams["sequence_length"],
            prediction_horizon=self.hparams["prediction_horizon"],
            lat_range=self.hparams["grid_lat_range"],
            lon_range=self.hparams["grid_lon_range"],
            grid_resolution=self.hparams["grid_resolution"],
            time_window=self.hparams["time_window"],
            stride=self.hparams["stride"],
        )
        self.train_dataset = IonosphereGraphCastDataset(train_df, **dataset_params)
        self.val_dataset = IonosphereGraphCastDataset(val_df, **dataset_params)
        self.test_dataset = IonosphereGraphCastDataset(test_df, **dataset_params)

    def _split_data(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        total_samples = len(self.dataframe)
        train_size, val_size = int(0.7 * total_samples), int(0.15 * total_samples)
        return (
            self.dataframe.slice(0, train_size),
            self.dataframe.slice(train_size, train_size + val_size),
            self.dataframe.slice(train_size + val_size, total_samples),
        )

    def _dataloader(self, dataset: IonosphereGraphCastDataset) -> Iterator:
        batch_size = self.hparams["batch_size"]
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            graph_inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            yield (graph_inputs, np.stack(targets))

    def train_dataloader(self) -> Iterator:
        return self._dataloader(self.train_dataset)

    def val_dataloader(self) -> Iterator:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> Iterator:
        return self._dataloader(self.test_dataset)
