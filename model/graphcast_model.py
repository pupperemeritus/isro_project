# model/graphcast_model.py
import dataclasses
from typing import Any, Callable, Mapping, Optional, Tuple

import chex
import haiku as hk
import jax.numpy as jnp
import jraph
import numpy as np
from graphcast import (
    deep_typed_graph_net,
    grid_mesh_connectivity,
    icosahedral_mesh,
    model_utils,
    sparse_transformer,
    transformer,
    typed_graph,
)
from scipy import sparse

Kwargs = Mapping[str, Any]
GNN = Callable[[jraph.GraphsTuple], jraph.GraphsTuple]


@chex.dataclass(frozen=True, eq=True)
class CompactGraphCastModelConfig:
    """Config for Compact GraphCast model tailored for Ionosphere data."""

    mesh_size: int = 1  # Reduced mesh size
    latent_size: int = 64  # Reduced latent size
    gnn_msg_steps: int = 4  # Reduced message passing steps
    hidden_layers: int = 1  # Reduced hidden layers
    resolution: Optional[int] = None  # Added to accept 'resolution' kwarg
    radius_query_fraction_edge_length: float = 0.8


@chex.dataclass(frozen=True, eq=True)
class CompactGraphCastTaskConfig:
    """Task config for Compact GraphCast, adjusted for ionosphere data."""

    input_variables: tuple[str, ...] = ("features_s4", "features_phase")
    target_variables: tuple[str, ...] = ("target_s4", "target_phase")
    pressure_levels: tuple[int, ...] = ()  # No pressure levels for ionosphere
    input_duration: str = "15m"  # Example duration


class CompactGraphCast(hk.Module):
    """Simplified GraphCast model for Ionosphere Forecasting."""

    def __init__(
        self,
        model_config: CompactGraphCastModelConfig,
        task_config: CompactGraphCastTaskConfig,
        name: str = "CompactGraphCast",
    ):
        super().__init__(name=name)
        self.model_config = model_config
        self.task_config = task_config

        self._spatial_features_kwargs = dict(
            add_node_positions=False,
            add_node_latitude=True,
            add_node_longitude=True,
            add_relative_positions=True,
            relative_longitude_local_coordinates=True,
            relative_latitude_local_coordinates=True,
        )

        self._meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
            splits=model_config.mesh_size
        )

        self._grid2mesh_gnn = deep_typed_graph_net.DeepTypedGraphNet(
            activation="swish",
            edge_latent_size=dict(grid2mesh=model_config.latent_size),
            embed_edges=True,
            embed_nodes=True,
            f32_aggregation=True,
            include_sent_messages_in_node_update=False,
            mlp_hidden_size=model_config.latent_size,
            mlp_num_hidden_layers=model_config.hidden_layers,
            name="grid2mesh_gnn",
            node_latent_size=dict(
                grid_nodes=model_config.latent_size,
                mesh_nodes=model_config.latent_size,
            ),
            node_output_size=None,  # Output size determined dynamically in _create_model
            num_message_passing_steps=1,
            use_layer_norm=True,
            use_norm_conditioning=False,  # Removed norm conditioning for simplicity
        )

        self._mesh_gnn = transformer.MeshTransformer(
            name="mesh_transformer",
            transformer_ctor=sparse_transformer.Transformer,
            transformer_kwargs={
                "attention_k_hop": 2,  # Example value, adjust as needed
                "d_model": model_config.latent_size,
                "num_layers": model_config.gnn_msg_steps,
                "num_heads": 4,  # Example value, adjust as needed
                "attention_type": "mha",  # Using standard MHA for simplicity
                "mask_type": "full",
            },
        )

        self._mesh2grid_gnn = deep_typed_graph_net.DeepTypedGraphNet(
            activation="swish",
            edge_latent_size=dict(mesh2grid=model_config.latent_size),
            embed_nodes=False,
            f32_aggregation=False,
            include_sent_messages_in_node_update=False,
            mlp_hidden_size=model_config.latent_size,
            mlp_num_hidden_layers=model_config.hidden_layers,
            name="mesh2grid_gnn",
            node_latent_size=dict(
                grid_nodes=model_config.latent_size,
                mesh_nodes=model_config.latent_size,
            ),
            node_output_size={"grid_nodes": 2},  # Outputting 2 features (s4, phase)
            num_message_passing_steps=1,
            use_layer_norm=True,
            use_norm_conditioning=False,  # Removed norm conditioning for simplicity
        )

        self._query_radius = (
            _get_max_edge_distance(self._meshes[-1])
            * model_config.radius_query_fraction_edge_length
        )
        self._initialized = False
        self._num_mesh_nodes = None
        self._mesh_nodes_lat = None
        self._mesh_nodes_lon = None
        self._grid_lat = None
        self._grid_lon = None
        self._num_grid_nodes = None
        self._grid_nodes_lat = None
        self._grid_nodes_lon = None
        self._grid2mesh_graph_structure = None
        self._mesh_graph_structure = None
        self._mesh2grid_graph_structure = None

    def __call__(self, graph: typed_graph.TypedGraph) -> typed_graph.TypedGraph:
        # Handle case where input is a list
        if isinstance(graph, list):
            graph = graph[0]

        self._maybe_init(graph)

        latent_mesh_nodes, latent_grid_nodes = self._run_grid2mesh_gnn(
            graph.nodes["grid_nodes"].features
        )
        updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)
        output_grid_nodes = self._run_mesh2grid_gnn(
            updated_latent_mesh_nodes, latent_grid_nodes
        )

        return graph._replace(
            nodes={
                "grid_nodes": graph.nodes["grid_nodes"]._replace(
                    features=output_grid_nodes
                )
            }
        )

    def _run_grid2mesh_gnn(
        self,
        grid_node_features: jnp.ndarray,
    ):
        batch_size = grid_node_features.shape[1]

        grid2mesh_graph = self._grid2mesh_graph_structure
        assert grid2mesh_graph is not None
        grid_nodes = grid2mesh_graph.nodes["grid_nodes"]
        mesh_nodes = grid2mesh_graph.nodes["mesh_nodes"]

        # Reshape grid_node_features to match the expected shape
        grid_node_features = jnp.transpose(
            grid_node_features, [1, 0, 2]
        )  # [batch, nodes, features]

        # Get spatial features and reshape them
        spatial_features = _add_batch_second_axis(
            grid_nodes.features.astype(grid_node_features.dtype), batch_size
        )  # [nodes, batch, features]
        spatial_features = jnp.transpose(
            spatial_features, [1, 0, 2]
        )  # [batch, nodes, features]

        # Combine features along the last dimension
        new_grid_nodes = grid_nodes._replace(
            features=jnp.concatenate([grid_node_features, spatial_features], axis=-1)
        )

        # Update mesh nodes similarly
        dummy_mesh_node_features = jnp.zeros(
            (batch_size, self._num_mesh_nodes, grid_node_features.shape[-1]),
            dtype=grid_node_features.dtype,
        )
        mesh_spatial_features = _add_batch_second_axis(
            mesh_nodes.features.astype(grid_node_features.dtype), batch_size
        )
        mesh_spatial_features = jnp.transpose(mesh_spatial_features, [1, 0, 2])

        new_mesh_nodes = mesh_nodes._replace(
            features=jnp.concatenate(
                [dummy_mesh_node_features, mesh_spatial_features], axis=-1
            )
        )

        # Update edges
        grid2mesh_edges_key = grid2mesh_graph.edge_key_by_name("grid2mesh")
        edges = grid2mesh_graph.edges[grid2mesh_edges_key]
        edge_features = _add_batch_second_axis(
            edges.features.astype(grid_node_features.dtype), batch_size
        )
        new_edges = edges._replace(features=edge_features)

        input_graph = self._grid2mesh_graph_structure._replace(
            edges={grid2mesh_edges_key: new_edges},
            nodes={"grid_nodes": new_grid_nodes, "mesh_nodes": new_mesh_nodes},
        )
        grid2mesh_out = self._grid2mesh_gnn(input_graph)
        latent_mesh_nodes = grid2mesh_out.nodes["mesh_nodes"].features
        latent_grid_nodes = grid2mesh_out.nodes["grid_nodes"].features
        return latent_mesh_nodes, latent_grid_nodes

    def _run_mesh_gnn(self, latent_mesh_nodes: chex.Array):

        batch_size = latent_mesh_nodes.shape[1]
        mesh_graph = self._mesh_graph_structure
        assert mesh_graph is not None
        mesh_edges_key = mesh_graph.edge_key_by_name("mesh")
        edges = mesh_graph.edges[mesh_edges_key]
        msg = (
            "The setup currently requires to only have one kind of edge in the"
            " mesh GNN."
        )
        assert len(mesh_graph.edges) == 1, msg
        new_edges = edges._replace(
            features=_add_batch_second_axis(
                edges.features.astype(latent_mesh_nodes.dtype), batch_size
            )
        )

        nodes = mesh_graph.nodes["mesh_nodes"]
        nodes = nodes._replace(features=latent_mesh_nodes)
        input_graph = mesh_graph._replace(
            edges={mesh_edges_key: new_edges}, nodes={"mesh_nodes": nodes}
        )

        return self._mesh_gnn(input_graph).nodes["mesh_nodes"].features

    def _run_mesh2grid_gnn(
        self, updated_latent_mesh_nodes: chex.Array, latent_grid_nodes: chex.Array
    ):
        batch_size = updated_latent_mesh_nodes.shape[1]

        mesh2grid_graph = self._mesh2grid_graph_structure
        assert mesh2grid_graph is not None
        mesh_nodes = mesh2grid_graph.nodes["mesh_nodes"]
        grid_nodes = mesh2grid_graph.nodes["grid_nodes"]
        new_mesh_nodes = mesh_nodes._replace(features=updated_latent_mesh_nodes)
        new_grid_nodes = grid_nodes._replace(features=latent_grid_nodes)
        mesh2grid_key = mesh2grid_graph.edge_key_by_name("mesh2grid")
        edges = mesh2grid_graph.edges[mesh2grid_key]

        new_edges = edges._replace(
            features=_add_batch_second_axis(
                edges.features.astype(latent_grid_nodes.dtype), batch_size
            )
        )

        input_graph = self._mesh2grid_graph_structure._replace(
            edges={mesh2grid_key: new_edges},
            nodes={"mesh_nodes": new_mesh_nodes, "grid_nodes": new_grid_nodes},
        )

        output_graph = self._mesh2grid_gnn(input_graph)
        output_grid_nodes = output_graph.nodes["grid_nodes"].features
        return output_grid_nodes

    def _maybe_init(self, sample_graph: typed_graph.TypedGraph):
        """Inits everything that has a dependency on the input coordinates."""
        if isinstance(sample_graph, list):
            sample_graph = sample_graph[0]

        # Get input features and shapes
        input_features = sample_graph.nodes["grid_nodes"].features
        self.input_feature_dim = input_features.shape[-1]  # Save feature dimension
        self.num_input_nodes = input_features.shape[0]  # Save number of nodes

        # Initialize mesh properties first
        self._init_mesh_properties()

        # Extract grid coordinates matching input size
        input_coords = sample_graph.nodes["grid_nodes"].features[0]  # [num_nodes, 2]
        grid_lat = input_coords[:, 1].reshape(-1)
        grid_lon = input_coords[:, 0].reshape(-1)

        self._init_grid_properties(grid_lat=grid_lat, grid_lon=grid_lon)

        # Initialize graph structures
        self._grid2mesh_graph_structure = self._init_grid2mesh_graph()
        self._mesh_graph_structure = self._init_mesh_graph()
        self._mesh2grid_graph_structure = self._init_mesh2grid_graph()
        self._initialized = True

    def _init_mesh_properties(self):
        """Inits static properties that have to do with mesh nodes."""
        self._num_mesh_nodes = self._meshes[-1].vertices.shape[0]
        mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
            self._meshes[-1].vertices[:, 0],
            self._meshes[-1].vertices[:, 1],
            self._meshes[-1].vertices[:, 2],
        )
        (
            mesh_nodes_lat,
            mesh_nodes_lon,
        ) = model_utils.spherical_to_lat_lon(phi=mesh_phi, theta=mesh_theta)
        # Convert to f32 to ensure the lat/lon features aren't in f64.
        self._mesh_nodes_lat = mesh_nodes_lat.astype(np.float32)
        self._mesh_nodes_lon = mesh_nodes_lon.astype(np.float32)

    def _init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray):
        """Inits static properties that have to do with grid nodes."""
        self._grid_lat = grid_lat  # Now directly using input lat/lon
        self._grid_lon = grid_lon  # Now directly using input lat/lon
        self._num_grid_nodes = grid_lat.shape[0] * grid_lon.shape[0]

        grid_nodes_lon, grid_nodes_lat = np.meshgrid(grid_lon, grid_lat)
        self._grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32)
        self._grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32)

    def _init_grid2mesh_graph(self) -> typed_graph.TypedGraph:
        """Build Grid2Mesh graph."""
        assert self._grid_lat is not None and self._grid_lon is not None
        (grid_indices, mesh_indices) = grid_mesh_connectivity.radius_query_indices(
            grid_latitude=self._grid_lat,
            grid_longitude=self._grid_lon,
            mesh=self._meshes[-1],
            radius=self._query_radius,
        )

        senders = grid_indices
        receivers = mesh_indices
        (senders_node_features, receivers_node_features, edge_features) = (
            model_utils.get_bipartite_graph_spatial_features(
                senders_node_lat=self._grid_nodes_lat,
                senders_node_lon=self._grid_nodes_lon,
                receivers_node_lat=self._mesh_nodes_lat,
                receivers_node_lon=self._mesh_nodes_lon,
                senders=senders,
                receivers=receivers,
                edge_normalization_factor=None,
                **self._spatial_features_kwargs,
            )
        )

        n_grid_node = np.array([self._num_grid_nodes])
        n_mesh_node = np.array([self._num_mesh_nodes])
        n_edge = np.array([mesh_indices.shape[0]])
        grid_node_set = typed_graph.NodeSet(
            n_node=n_grid_node, features=senders_node_features
        )
        mesh_node_set = typed_graph.NodeSet(
            n_node=n_mesh_node, features=receivers_node_features
        )
        edge_set = typed_graph.EdgeSet(
            n_edge=n_edge,
            indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
            features=edge_features,
        )
        nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set}
        edges = {
            typed_graph.EdgeSetKey("grid2mesh", ("grid_nodes", "mesh_nodes")): edge_set
        }
        grid2mesh_graph = typed_graph.TypedGraph(
            context=typed_graph.Context(n_graph=np.array([1]), features=()),
            nodes=nodes,
            edges=edges,
        )
        return grid2mesh_graph

    def _init_mesh_graph(self) -> typed_graph.TypedGraph:
        """Build Mesh graph."""
        merged_mesh = icosahedral_mesh.merge_meshes(self._meshes)
        senders, receivers = icosahedral_mesh.faces_to_edges(merged_mesh.faces)
        assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
        node_features, edge_features = model_utils.get_graph_spatial_features(
            node_lat=self._mesh_nodes_lat,
            node_lon=self._mesh_nodes_lon,
            senders=senders,
            receivers=receivers,
            **self._spatial_features_kwargs,
        )

        n_mesh_node = np.array([self._num_mesh_nodes])
        n_edge = np.array([senders.shape[0]])
        assert n_mesh_node == len(node_features)
        mesh_node_set = typed_graph.NodeSet(n_node=n_mesh_node, features=node_features)
        edge_set = typed_graph.EdgeSet(
            n_edge=n_edge,
            indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
            features=edge_features,
        )
        nodes = {"mesh_nodes": mesh_node_set}
        edges = {typed_graph.EdgeSetKey("mesh", ("mesh_nodes", "mesh_nodes")): edge_set}
        mesh_graph = typed_graph.TypedGraph(
            context=typed_graph.Context(n_graph=np.array([1]), features=()),
            nodes=nodes,
            edges=edges,
        )
        return mesh_graph

    def _init_mesh2grid_graph(self) -> typed_graph.TypedGraph:
        """Build Mesh2Grid graph."""
        (grid_indices, mesh_indices) = grid_mesh_connectivity.in_mesh_triangle_indices(
            grid_latitude=self._grid_lat,
            grid_longitude=self._grid_lon,
            mesh=self._meshes[-1],
        )

        senders = mesh_indices
        receivers = grid_indices
        assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
        (senders_node_features, receivers_node_features, edge_features) = (
            model_utils.get_bipartite_graph_spatial_features(
                senders_node_lat=self._mesh_nodes_lat,
                senders_node_lon=self._mesh_nodes_lon,
                receivers_node_lat=self._grid_nodes_lat,
                receivers_node_lon=self._grid_nodes_lon,
                senders=senders,
                receivers=receivers,
                edge_normalization_factor=None,  # Removed normalization factor
                **self._spatial_features_kwargs,
            )
        )

        n_grid_node = np.array([self._num_grid_nodes])
        n_mesh_node = np.array([self._num_mesh_nodes])
        n_edge = np.array([senders.shape[0]])
        grid_node_set = typed_graph.NodeSet(
            n_node=n_grid_node, features=receivers_node_features
        )
        mesh_node_set = typed_graph.NodeSet(
            n_node=n_mesh_node, features=senders_node_features
        )
        edge_set = typed_graph.EdgeSet(
            n_edge=n_edge,
            indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
            features=edge_features,
        )
        nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set}
        edges = {
            typed_graph.EdgeSetKey("mesh2grid", ("mesh_nodes", "grid_nodes")): edge_set
        }
        mesh2grid_graph = typed_graph.TypedGraph(
            context=typed_graph.Context(n_graph=np.array([1]), features=()),
            nodes=nodes,
            edges=edges,
        )
        return mesh2grid_graph


def _get_max_edge_distance(mesh):
    senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces)
    edge_distances = np.linalg.norm(
        mesh.vertices[senders] - mesh.vertices[receivers], axis=-1
    )
    return edge_distances.max()


def _add_batch_second_axis(data, batch_size):
    # data [leading_dim, trailing_dim]
    assert data.ndim == 2
    ones = jnp.ones([batch_size, 1], dtype=data.dtype)
    return data[:, None] * ones  # [leading_dim, batch, trailing_dim]
