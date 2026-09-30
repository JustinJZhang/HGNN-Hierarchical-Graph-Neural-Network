import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import ACTION_UNIT_GRAPH, EXPRESSION_GRAPH, FACIAL_ASYMMETRY_GRAPH

DROP_OUT_RATE = 0.15
SCALE = 0.5


def apply_cross_constraint(logits):
    constrained_logits = logits.clone()
    absence_mask = torch.argmax(logits[:, 0, :], dim=1) == 0
    constrained_logits[absence_mask, 1, :] = logits.new_tensor(
        (1.0, 0.0, 0.0)
    )
    return constrained_logits


class GGA(nn.Module):
    """Apply graph-level context to node features through node-wise attention.

    Input: node features with shape [batch_size * num_nodes, num_features].
    Output: reweighted node features with the same shape as the input.
    """

    def __init__(self, num_nodes, ratio=4):
        super(GGA, self).__init__()

        self.num_nodes = num_nodes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_nodes = max(num_nodes // ratio, 1)
        self.fc1 = nn.Conv2d(num_nodes, hidden_nodes, 1, bias=False)
        self.norm = nn.LayerNorm((hidden_nodes, 1, 1))
        self.fc2 = nn.Conv2d(hidden_nodes, num_nodes, 1, bias=False)

    def forward(self, x, batch_size):
        num_features = x.shape[1]
        x = x.reshape(batch_size, self.num_nodes, num_features, 1)

        avg_out = self.fc2(F.gelu(self.norm(self.fc1(self.avg_pool(x)))))
        max_out = self.fc2(F.gelu(self.norm(self.fc1(self.max_pool(x)))))
        attention = torch.sigmoid(avg_out + max_out)
        x = attention * x

        return x.reshape(batch_size * self.num_nodes, num_features)


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc0 = nn.Linear(int(512 * SCALE), int(256 * SCALE))
        self.norm0 = nn.LayerNorm(int(256 * SCALE))
        self.fc1 = nn.Linear(int(256 * SCALE), int(128 * SCALE))
        self.norm1 = nn.LayerNorm(int(128 * SCALE))
        self.fc2 = nn.Linear(int(128 * SCALE), 3)
        self.dropout = nn.Dropout(p=DROP_OUT_RATE)

    def forward(self, x1, x2):
        x_ori = torch.cat((x1, x2), dim=1)
        x = F.gelu(self.norm0(self.fc0(x_ori)))
        x = self.dropout(x)
        x = F.gelu(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Projector(nn.Module):
    def __init__(self, input_size, output_size):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_size, int(input_size / 2))
        self.norm1 = nn.LayerNorm(int(input_size / 2))
        self.fc2 = nn.Linear(int(input_size / 2), int(input_size / 4))
        self.norm2 = nn.LayerNorm(int(input_size / 4))
        self.fc3 = nn.Linear(int(input_size / 4), output_size)
        self.dropout = nn.Dropout(p=DROP_OUT_RATE)

    def forward(self, x):
        x = F.gelu(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.gelu(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Projector_exp(nn.Module):
    def __init__(self, input_size, output_size):
        super(Projector_exp, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, int(input_size))
        self.norm1 = nn.LayerNorm(int(input_size))
        self.fc2 = nn.Linear(int(input_size), int(input_size / 2))
        self.norm2 = nn.LayerNorm(int(input_size / 2))
        self.fc3 = nn.Linear(int(input_size / 2), output_size)
        self.dropout = nn.Dropout(p=DROP_OUT_RATE)

    def forward(self, x_1, x_2):
        x_ori = torch.cat((x_1, x_2), dim=1)
        x = F.gelu(self.norm1(self.fc1(x_ori)))
        x = self.dropout(x)
        x = F.gelu(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class GraphConvolution(nn.Module):
    """Map node features [N, C_in] and edges [2, E] to features [N, C_out]."""

    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(2 * in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        source, destination = edge_index
        neighbor_edges = source != destination
        source = source[neighbor_edges]
        destination = destination[neighbor_edges]

        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.index_add_(0, destination, x[source])
        neighbor_count = x.new_zeros((x.shape[0], 1))
        neighbor_count.index_add_(
            0, destination, x.new_ones((destination.numel(), 1))
        )
        neighbor_mean = neighbor_sum / neighbor_count.clamp_min(1)

        combined = torch.cat((neighbor_mean, x), dim=1)
        return F.gelu(self.norm(self.linear(combined)))


class Expression_Path(nn.Module):
    def __init__(self, num_AU, num_of_fa_feat, num_facial_expressions, device):
        super(Expression_Path, self).__init__()

        self.gcn1_au = GraphConvolution(num_AU, int(64 * SCALE))
        self.att1_au = GGA(num_facial_expressions)
        self.gcn2_au = GraphConvolution(int(64 * SCALE), int(128 * SCALE))
        self.att2_au = GGA(num_facial_expressions)
        self.gcn3_au = GraphConvolution(int(128 * SCALE), int(256 * SCALE))
        self.att3_au = GGA(num_facial_expressions)

        self.gcn1_fa_dev = GraphConvolution(10, int(64 * SCALE))
        self.att1_fa_dev = GGA(num_facial_expressions)
        self.gcn2_fa_dev = GraphConvolution(int(64 * SCALE), int(128 * SCALE))
        self.att2_fa_dev = GGA(num_facial_expressions)
        self.gcn3_fa_dev = GraphConvolution(int(128 * SCALE), int(256 * SCALE))
        self.att3_fa_dev = GGA(num_facial_expressions)

        self.gcn1_fa_lr = GraphConvolution(14, int(64 * SCALE))
        self.att1_fa_lr = GGA(num_facial_expressions)
        self.gcn2_fa_lr = GraphConvolution(int(64 * SCALE), int(128 * SCALE))
        self.att2_fa_lr = GGA(num_facial_expressions)
        self.gcn3_fa_lr = GraphConvolution(int(128 * SCALE), int(256 * SCALE))
        self.att3_fa_lr = GGA(num_facial_expressions)

        self.edge_list_expression = EXPRESSION_GRAPH.edges

        self.severity_fcn = Projector_exp(int((256 + 128 + 64) * 8 * SCALE), int(256 * SCALE))
        self.asymmetry_fcn = Projector_exp(int((256 + 128 + 64) * 8 * SCALE), int(256 * SCALE))

        self.num_AU = num_AU
        self.num_of_fa_feat = num_of_fa_feat
        self.num_facial_expressions = num_facial_expressions
        self.device = device

    def forward(self, x_au, x_fa_dev, x_fa_lr, batch_size):

        adjusted_edge_list = []
        for batch_idx in range(batch_size):
            offset = batch_idx * self.num_facial_expressions
            for edge in self.edge_list_expression:
                adjusted_edge_list.append([edge[0] + offset, edge[1] + offset])
        edge_index_expression = torch.tensor(adjusted_edge_list, dtype=torch.long).t().contiguous().to(self.device)

        x_au = self.gcn1_au(x_au, edge_index_expression)
        x_au_1 = self.att1_au(x_au, batch_size)
        x_au_2 = self.gcn2_au(x_au_1, edge_index_expression)
        x_au_2 = self.att2_au(x_au_2, batch_size)
        x_au_3 = self.gcn3_au(x_au_2, edge_index_expression)
        x_au_3 = self.att3_au(x_au_3, batch_size)

        x_fa_dev = self.gcn1_fa_dev(x_fa_dev, edge_index_expression)
        x_fa_dev_1 = self.att1_fa_dev(x_fa_dev, batch_size)
        x_fa_dev_2 = self.gcn2_fa_dev(x_fa_dev_1, edge_index_expression)
        x_fa_dev_2 = self.att2_fa_dev(x_fa_dev_2, batch_size)
        x_fa_dev_3 = self.gcn3_fa_dev(x_fa_dev_2, edge_index_expression)
        x_fa_dev_3 = self.att3_fa_dev(x_fa_dev_3, batch_size)

        x_fa_lr = self.gcn1_fa_lr(x_fa_lr, edge_index_expression)
        x_fa_lr_1 = self.att1_fa_lr(x_fa_lr, batch_size)
        x_fa_lr_2 = self.gcn2_fa_lr(x_fa_lr_1, edge_index_expression)
        x_fa_lr_2 = self.att2_fa_lr(x_fa_lr_2, batch_size)
        x_fa_lr_3 = self.gcn3_fa_lr(x_fa_lr_2, edge_index_expression)
        x_fa_lr_3 = self.att3_fa_lr(x_fa_lr_3, batch_size)

        x_au_1 = x_au_1.reshape(batch_size, -1)
        x_au_2 = x_au_2.reshape(batch_size, -1)
        x_au_3 = x_au_3.reshape(batch_size, -1)
        x_fa_dev_1 = x_fa_dev_1.reshape(batch_size, -1)
        x_fa_dev_2 = x_fa_dev_2.reshape(batch_size, -1)
        x_fa_dev_3 = x_fa_dev_3.reshape(batch_size, -1)
        x_fa_lr_1 = x_fa_lr_1.reshape(batch_size, -1)
        x_fa_lr_2 = x_fa_lr_2.reshape(batch_size, -1)
        x_fa_lr_3 = x_fa_lr_3.reshape(batch_size, -1)

        x_au = torch.cat((x_au_1, x_au_2, x_au_3), dim=1)
        x_fa_dev = torch.cat((x_fa_dev_1, x_fa_dev_2, x_fa_dev_3), dim=1)
        x_fa_lr = torch.cat((x_fa_lr_1, x_fa_lr_2, x_fa_lr_3), dim=1)

        severity_output = self.severity_fcn(x_au, x_fa_dev)
        asymmetry_output = self.asymmetry_fcn(x_fa_lr, x_fa_dev)

        return severity_output, asymmetry_output


class AU_Path(nn.Module):
    def __init__(self, num_AU, num_of_fa_feat, num_facial_expressions, device):
        super(AU_Path, self).__init__()

        self.gcn1_au = GraphConvolution(num_facial_expressions, int(64 * SCALE))
        self.att1_au = GGA(num_AU)
        self.gcn2_au = GraphConvolution(int(64 * SCALE), int(128 * SCALE))
        self.att2_au = GGA(num_AU)
        self.gcn3_au = GraphConvolution(int(128 * SCALE), int(256 * SCALE))
        self.att3_au = GGA(num_AU)

        self.edge_list_au = ACTION_UNIT_GRAPH.edges

        self.severity_fcn = Projector(int(17 * (256 + 128 + 64) * SCALE), int(256 * SCALE))

        self.num_AU = num_AU
        self.num_facial_expressions = num_facial_expressions
        self.device = device

    def forward(self, x_au, batch_size):
        x_au = x_au.reshape(batch_size, self.num_facial_expressions, self.num_AU)
        x_au = x_au.permute(0, 2, 1)
        x_au = x_au.reshape(batch_size * self.num_AU, self.num_facial_expressions)

        adjusted_edge_list = []
        for batch_idx in range(batch_size):
            offset = batch_idx * self.num_AU
            for edge in self.edge_list_au:
                adjusted_edge_list.append([edge[0] + offset, edge[1] + offset])
        edge_index_au = torch.tensor(adjusted_edge_list, dtype=torch.long).t().contiguous().to(self.device)

        x_au = self.gcn1_au(x_au, edge_index_au)
        x_au_1 = self.att1_au(x_au, batch_size)
        x_au_2 = self.gcn2_au(x_au_1, edge_index_au)
        x_au_2 = self.att2_au(x_au_2, batch_size)
        x_au_3 = self.gcn3_au(x_au_2, edge_index_au)
        x_au_3 = self.att3_au(x_au_3, batch_size)

        x_au_1 = x_au_1.reshape(batch_size, -1)
        x_au_2 = x_au_2.reshape(batch_size, -1)
        x_au_3 = x_au_3.reshape(batch_size, -1)
        x_au = torch.cat((x_au_1, x_au_2, x_au_3), dim=1)
        severity_output = self.severity_fcn(x_au)

        return severity_output


class FA_Path(nn.Module):
    def __init__(self, num_AU, num_of_fa_feat, num_facial_expressions, device):
        super(FA_Path, self).__init__()

        self.gcn1_fa = GraphConvolution(num_facial_expressions, int(64 * SCALE))
        self.att1_fa = GGA(num_of_fa_feat)
        self.gcn2_fa = GraphConvolution(int(64 * SCALE), int(128 * SCALE))
        self.att2_fa = GGA(num_of_fa_feat)
        self.gcn3_fa = GraphConvolution(int(128 * SCALE), int(256 * SCALE))
        self.att3_fa = GGA(num_of_fa_feat)

        self.edge_list_fa = FACIAL_ASYMMETRY_GRAPH.edges

        self.asymmetry_fcn = Projector(int(24 * (256 + 128 + 64) * SCALE), int(256 * SCALE))

        self.num_AU = num_AU
        self.num_of_fa_feat = num_of_fa_feat
        self.num_facial_expressions = num_facial_expressions
        self.device = device

    def forward(self, x_fa, batch_size):
        x_fa = x_fa.reshape(batch_size, self.num_facial_expressions, self.num_of_fa_feat)
        x_fa = x_fa.permute(0, 2, 1)
        x_fa = x_fa.reshape(batch_size * self.num_of_fa_feat, self.num_facial_expressions)

        adjusted_edge_list = []
        for batch_idx in range(batch_size):
            offset = batch_idx * self.num_of_fa_feat
            for edge in self.edge_list_fa:
                adjusted_edge_list.append([edge[0] + offset, edge[1] + offset])
        edge_index_fa = torch.tensor(adjusted_edge_list, dtype=torch.long).t().contiguous().to(self.device)

        x_fa = self.gcn1_fa(x_fa, edge_index_fa)
        x_fa_1 = self.att1_fa(x_fa, batch_size)
        x_fa_2 = self.gcn2_fa(x_fa_1, edge_index_fa)
        x_fa_2 = self.att2_fa(x_fa_2, batch_size)
        x_fa_3 = self.gcn3_fa(x_fa_2, edge_index_fa)
        x_fa_3 = self.att3_fa(x_fa_3, batch_size)

        x_fa_1 = x_fa_1.reshape(batch_size, -1)
        x_fa_2 = x_fa_2.reshape(batch_size, -1)
        x_fa_3 = x_fa_3.reshape(batch_size, -1)
        x_fa = torch.cat((x_fa_1, x_fa_2, x_fa_3), dim=1)
        asymmetry_output = self.asymmetry_fcn(x_fa)
        return asymmetry_output


class HGNN(nn.Module):
    def __init__(self, num_AU, num_of_fa_feat, num_facial_expressions, device):
        super(HGNN, self).__init__()

        self.expression_path = Expression_Path(num_AU, num_of_fa_feat, num_facial_expressions, device)
        self.au_path = AU_Path(num_AU, num_of_fa_feat, num_facial_expressions, device)
        self.fa_path = FA_Path(num_AU, num_of_fa_feat, num_facial_expressions, device)

        self.weightedpred_severity = Predictor()
        self.weightedpred_asymmetry = Predictor()

        self.num_AU = num_AU
        self.num_of_fa_feat = num_of_fa_feat
        self.num_facial_expressions = num_facial_expressions
        self.device = device
        self.use_cross_constraint = True

    def forward(self, x):
        x_au, x_fa = torch.split(x, [self.num_AU, x.size(1) - self.num_AU], dim=1)
        x_fa_dev, x_fa_lr = torch.split(x_fa, [10, 14], dim=1)

        batch_size = x_au.size(0) // self.num_facial_expressions

        severity_expression, asymmetry_expression = self.expression_path(x_au, x_fa_dev, x_fa_lr, batch_size)
        severity_au = self.au_path(x_au, batch_size)
        asymmetry_fa = self.fa_path(x_fa, batch_size)

        severity = self.weightedpred_severity(severity_expression, severity_au)
        asymmetry = self.weightedpred_asymmetry(asymmetry_expression, asymmetry_fa)

        combined_output = torch.stack([severity, asymmetry], dim=1)
        return apply_cross_constraint(combined_output) if self.use_cross_constraint else combined_output
