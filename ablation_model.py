from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from graph import ACTION_UNIT_GRAPH, EXPRESSION_GRAPH, FACIAL_ASYMMETRY_GRAPH
from model import (
    GGA,
    GraphConvolution,
    Predictor,
    Projector,
    Projector_exp,
    SCALE,
    apply_cross_constraint,
)


@dataclass(frozen=True)
class AblationDefinition:
    """Define one HGNN ablation.

    Attributes:
        use_expression_graph: Whether to use the expression graph G_e.
        use_action_graph: Whether to use the action graph G_au.
        use_asymmetry_graph: Whether to use the asymmetry graph G_fa.
        num_gcn_layers: Number of GCN+GGA layers, from 1 to 3.
        use_gga: Whether to apply global graph attention.
        use_cross_constraint: Whether to apply the cross constraint.
    """

    use_expression_graph: bool
    use_action_graph: bool
    use_asymmetry_graph: bool
    num_gcn_layers: int = 3
    use_gga: bool = True
    use_cross_constraint: bool = True


GRAPH_ABLATIONS: Dict[str, AblationDefinition] = {
    "ge": AblationDefinition(True, False, False),
    "ge_gau": AblationDefinition(True, True, False),
    "ge_gfa": AblationDefinition(True, False, True),
    "gau_gfa": AblationDefinition(False, True, True),
    "ge_gau_gfa": AblationDefinition(True, True, True),
}


MODULE_ABLATIONS: Dict[str, AblationDefinition] = {
    "without_gga": AblationDefinition(
        True,
        True,
        True,
        use_gga=False,
    ),
    "without_cross_constraint": AblationDefinition(
        True,
        True,
        True,
        use_cross_constraint=False,
    ),
}

DEPTH_ABLATIONS: Dict[str, AblationDefinition] = {
    "depth_1": AblationDefinition(True, True, True, num_gcn_layers=1),
    "depth_2": AblationDefinition(True, True, True, num_gcn_layers=2),
    "depth_3": AblationDefinition(True, True, True, num_gcn_layers=3),
}

ABLATION_DEFINITIONS: Dict[str, AblationDefinition] = {
    **GRAPH_ABLATIONS,
    **MODULE_ABLATIONS,
    **DEPTH_ABLATIONS,
}


class ConfigurableGraphEncoder(nn.Module):
    """Encode one graph with 1-3 GCN+GGA layers.

    Args:
        input_size: Integer input feature count per graph node.
        num_nodes: Integer node count in the active graph.
        edges: Directed graph edges with shape [num_edges, 2].
        num_layers: Integer number of layers, from 1 to 3.
        use_gga: Boolean controlling GGA after each GCN layer.
        device: PyTorch device on which batched graph edges are constructed.

    Input:
        Floating-point node tensor with shape
        [batch_size * num_nodes, input_size].

    Output:
        Floating-point graph embedding with shape
        [batch_size, num_nodes * sum(layer_output_sizes)].
    """

    LAYER_SIZES = (
        int(64 * SCALE),
        int(128 * SCALE),
        int(256 * SCALE),
    )

    def __init__(
        self,
        input_size,
        num_nodes,
        edges,
        num_layers,
        use_gga,
        device,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.edges = tuple(edges)
        self.num_layers = num_layers
        self.use_gga = use_gga
        self.device = device

        layer_input_sizes = (input_size,) + self.LAYER_SIZES[:-1]
        self.gcn_layers = nn.ModuleList(
            GraphConvolution(layer_input_sizes[index], self.LAYER_SIZES[index])
            for index in range(num_layers)
        )
        self.gga_layers = (
            nn.ModuleList(GGA(num_nodes) for _ in range(num_layers))
            if use_gga
            else nn.ModuleList()
        )

    @property
    def output_size(self):
        """Return the flattened graph embedding dimension as an integer."""
        return self.num_nodes * sum(self.LAYER_SIZES[: self.num_layers])

    def _build_edge_index(self, batch_size):
        """Build a batched directed edge tensor.

        Args:
            batch_size: Integer number of graphs in the batch.

        Returns:
            Long tensor with shape [2, batch_size * num_edges].
        """
        batched_edges = []
        for batch_index in range(batch_size):
            offset = batch_index * self.num_nodes
            for source, target in self.edges:
                batched_edges.append((source + offset, target + offset))
        return torch.tensor(
            batched_edges,
            dtype=torch.long,
            device=self.device,
        ).t().contiguous()

    def forward(self, x, batch_size):
        """Encode and concatenate all selected layer outputs.

        Args:
            x: Floating-point tensor with shape
                [batch_size * num_nodes, input_size].
            batch_size: Integer number of graphs in the batch.

        Returns:
            Floating-point tensor with shape [batch_size, output_size].
        """
        edge_index = self._build_edge_index(batch_size)
        layer_outputs = []
        for index, graph_convolution in enumerate(self.gcn_layers):
            x = graph_convolution(x, edge_index)
            if self.use_gga:
                x = self.gga_layers[index](x, batch_size)
            layer_outputs.append(x.reshape(batch_size, -1))
        return torch.cat(layer_outputs, dim=1)


class ConfigurableExpressionPath(nn.Module):
    """Encode the three expression-graph branches.

    Args:
        num_AU: Integer action-unit feature count; expected value 17.
        num_facial_expressions: Integer expression count; expected value 8.
        num_layers: Integer number of GCN+GGA layers, from 1 to 3.
        use_gga: Boolean controlling GGA after every GCN layer.
        device: PyTorch device used for graph construction and computation.

    Inputs:
        AU tensor [batch_size * 8, 17], deviation tensor
        [batch_size * 8, 10], unilateral tensor [batch_size * 8, 14], and an
        integer batch size.

    Outputs:
        Severity and asymmetry embeddings, each with shape [batch_size, 128].
    """

    def __init__(
        self,
        num_AU,
        num_facial_expressions,
        num_layers,
        use_gga,
        device,
    ):
        super().__init__()
        encoder_arguments = (
            num_facial_expressions,
            EXPRESSION_GRAPH.edges,
            num_layers,
            use_gga,
            device,
        )
        self.au_encoder = ConfigurableGraphEncoder(num_AU, *encoder_arguments)
        self.deviation_encoder = ConfigurableGraphEncoder(10, *encoder_arguments)
        self.unilateral_encoder = ConfigurableGraphEncoder(14, *encoder_arguments)

        projector_input_size = self.au_encoder.output_size
        projector_output_size = int(256 * SCALE)
        self.severity_projector = Projector_exp(
            projector_input_size,
            projector_output_size,
        )
        self.asymmetry_projector = Projector_exp(
            projector_input_size,
            projector_output_size,
        )

    def forward(self, x_au, x_fa_dev, x_fa_lr, batch_size):
        """Return expression-graph embeddings.

        Args:
            x_au: Floating-point AU tensor with shape [batch_size * 8, 17].
            x_fa_dev: Floating-point deviation tensor with shape
                [batch_size * 8, 10].
            x_fa_lr: Floating-point unilateral tensor with shape
                [batch_size * 8, 14].
            batch_size: Integer number of subjects.

        Returns:
            Tuple of severity and asymmetry tensors, each [batch_size, 128].
        """
        au_embedding = self.au_encoder(x_au, batch_size)
        deviation_embedding = self.deviation_encoder(x_fa_dev, batch_size)
        unilateral_embedding = self.unilateral_encoder(x_fa_lr, batch_size)
        severity = self.severity_projector(au_embedding, deviation_embedding)
        asymmetry = self.asymmetry_projector(
            unilateral_embedding,
            deviation_embedding,
        )
        return severity, asymmetry


class ConfigurableNodePath(nn.Module):
    """Encode the action graph or facial-asymmetry graph.

    Args:
        num_nodes: Integer graph node count: 17 for G_au or 24 for G_fa.
        num_facial_expressions: Integer input feature count per node; expected 8.
        edges: Directed graph edges with shape [num_edges, 2].
        num_layers: Integer number of GCN+GGA layers, from 1 to 3.
        use_gga: Boolean controlling GGA after every GCN layer.
        device: PyTorch device used for graph construction and computation.

    Input:
        Floating-point expression-major tensor with shape
        [batch_size * num_facial_expressions, num_nodes].

    Output:
        Floating-point projected embedding with shape [batch_size, 128].
    """

    def __init__(
        self,
        num_nodes,
        num_facial_expressions,
        edges,
        num_layers,
        use_gga,
        device,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_facial_expressions = num_facial_expressions
        self.encoder = ConfigurableGraphEncoder(
            num_facial_expressions,
            num_nodes,
            edges,
            num_layers,
            use_gga,
            device,
        )
        self.projector = Projector(
            self.encoder.output_size,
            int(256 * SCALE),
        )

    def forward(self, x, batch_size):
        """Return a graph embedding.

        Args:
            x: Floating-point tensor with shape
                [batch_size * num_facial_expressions, num_nodes].
            batch_size: Integer number of subjects.

        Returns:
            Floating-point tensor with shape [batch_size, 128].
        """
        x = x.reshape(
            batch_size,
            self.num_facial_expressions,
            self.num_nodes,
        )
        x = x.permute(0, 2, 1).reshape(
            batch_size * self.num_nodes,
            self.num_facial_expressions,
        )
        return self.projector(self.encoder(x, batch_size))


class AblationHGNN(nn.Module):
    """Implement graph, module, and depth ablations for HGNN.

    Args:
        num_AU: Integer action-unit feature count; expected value 17.
        num_of_fa_feat: Integer facial-asymmetry feature count; expected value 24.
        num_facial_expressions: Integer expression count; expected value 8.
        device: PyTorch device used for graph construction and computation.
        variant: String key in ``ABLATION_DEFINITIONS``.

    Input:
        Floating-point tensor with shape [batch_size * 8, 41].

    Output:
        Floating-point logits with shape [batch_size, 2, 3].
    """

    def __init__(
        self,
        num_AU,
        num_of_fa_feat,
        num_facial_expressions,
        device,
        variant,
    ):
        super().__init__()
        if variant not in ABLATION_DEFINITIONS:
            valid_variants = ", ".join(ABLATION_DEFINITIONS)
            raise ValueError(
                f"Unknown ablation variant: {variant}. Expected one of: "
                f"{valid_variants}."
            )

        self.definition = ABLATION_DEFINITIONS[variant]
        self.variant = variant
        self.num_AU = num_AU
        self.num_of_fa_feat = num_of_fa_feat
        self.num_facial_expressions = num_facial_expressions
        self.embedding_size = int(256 * SCALE)

        path_arguments = (
            self.definition.num_gcn_layers,
            self.definition.use_gga,
            device,
        )
        self.expression_path = (
            ConfigurableExpressionPath(
                num_AU,
                num_facial_expressions,
                *path_arguments,
            )
            if self.definition.use_expression_graph
            else None
        )
        self.au_path = (
            ConfigurableNodePath(
                num_AU,
                num_facial_expressions,
                ACTION_UNIT_GRAPH.edges,
                *path_arguments,
            )
            if self.definition.use_action_graph
            else None
        )
        self.fa_path = (
            ConfigurableNodePath(
                num_of_fa_feat,
                num_facial_expressions,
                FACIAL_ASYMMETRY_GRAPH.edges,
                *path_arguments,
            )
            if self.definition.use_asymmetry_graph
            else None
        )

        self.weightedpred_severity = Predictor()
        self.weightedpred_asymmetry = Predictor()

    def forward(self, x):
        """Return severity and asymmetry logits.

        Args:
            x: Floating-point tensor with shape [batch_size * 8, 41].

        Returns:
            Floating-point logits with shape [batch_size, 2, 3]. Removed graphs
            contribute zero embeddings with shape [batch_size, 128].
        """
        x_au, x_fa = torch.split(
            x,
            [self.num_AU, self.num_of_fa_feat],
            dim=1,
        )
        x_fa_dev, x_fa_lr = torch.split(x_fa, [10, 14], dim=1)
        batch_size = x.shape[0] // self.num_facial_expressions
        zero_embedding = x.new_zeros((batch_size, self.embedding_size))

        if self.expression_path is None:
            severity_expression = zero_embedding
            asymmetry_expression = zero_embedding
        else:
            severity_expression, asymmetry_expression = self.expression_path(
                x_au,
                x_fa_dev,
                x_fa_lr,
                batch_size,
            )

        severity_action = (
            zero_embedding
            if self.au_path is None
            else self.au_path(x_au, batch_size)
        )
        asymmetry_fa = (
            zero_embedding
            if self.fa_path is None
            else self.fa_path(x_fa, batch_size)
        )

        severity = self.weightedpred_severity(
            severity_expression,
            severity_action,
        )
        asymmetry = self.weightedpred_asymmetry(
            asymmetry_expression,
            asymmetry_fa,
        )
        logits = torch.stack((severity, asymmetry), dim=1)
        if self.definition.use_cross_constraint:
            logits = apply_cross_constraint(logits)
        return logits


def build_ablation_model(
    variant,
    num_AU,
    num_of_fa_feat,
    num_facial_expressions,
    device,
):
    """Build one configured HGNN ablation model.

    Args:
        variant: String key in ``ABLATION_DEFINITIONS``.
        num_AU: Integer action-unit feature count; expected value 17.
        num_of_fa_feat: Integer facial-asymmetry feature count; expected value 24.
        num_facial_expressions: Integer expression count; expected value 8.
        device: PyTorch device used for graph construction and computation.

    Returns:
        ``AblationHGNN`` mapping input tensors [batch_size * 8, 41] to logits
        [batch_size, 2, 3].
    """
    return AblationHGNN(
        num_AU,
        num_of_fa_feat,
        num_facial_expressions,
        device,
        variant,
    )
