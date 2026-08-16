import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from graph import ACTION_UNIT_GRAPH, EXPRESSION_GRAPH, FACIAL_ASYMMETRY_GRAPH

DROP_OUT_RATE = 0.15
SCALE = 0.5


class GGA(nn.Module):
    def __init__(self, in_planes, ratio=4, learn_W_ori=False):
        super(GGA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        if learn_W_ori:
            self.w_ori = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        else:
            self.w_ori = 0.3

    def forward(self, x):
        x_ori = x
        avg_out = self.fc2(F.gelu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.gelu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        x = torch.sigmoid(out) * x + x_ori * self.w_ori
        return x


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc0 = nn.Linear(int(512 * SCALE), int(256 * SCALE))
        self.bn0 = nn.LayerNorm(int(256 * SCALE))
        self.fc1 = nn.Linear(int(256 * SCALE), int(128 * SCALE))
        self.bn1 = nn.LayerNorm(int(128 * SCALE))
        self.fc2 = nn.Linear(int(128 * SCALE), 3)
        self.dropout = nn.Dropout(p=DROP_OUT_RATE)

    def forward(self, x1, x2):
        x_ori = torch.cat((x1, x2), dim=1)
        x = F.gelu(self.bn0(self.fc0(x_ori)))
        x = self.dropout(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Projector(nn.Module):
    def __init__(self, input_size, output_size):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_size, int(input_size / 2))
        self.bn1 = nn.LayerNorm(int(input_size / 2))
        self.fc2 = nn.Linear(int(input_size / 2), int(input_size / 4))
        self.bn2 = nn.LayerNorm(int(input_size / 4))
        self.fc3 = nn.Linear(int(input_size / 4), output_size)
        self.dropout = nn.Dropout(p=DROP_OUT_RATE)

    def forward(self, x):
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Projector_exp(nn.Module):
    def __init__(self, input_size, output_size):
        super(Projector_exp, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, int(input_size))
        self.bn1 = nn.LayerNorm(int(input_size))
        self.fc2 = nn.Linear(int(input_size), int(input_size / 2))
        self.bn2 = nn.LayerNorm(int(input_size / 2))
        self.fc3 = nn.Linear(int(input_size / 2), output_size)
        self.dropout = nn.Dropout(p=DROP_OUT_RATE)

    def forward(self, x_1, x_2):
        x_ori = torch.cat((x_1, x_2), dim=1)
        x = F.gelu(self.bn1(self.fc1(x_ori)))
        x = self.dropout(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class EdgeWeightPredictor(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        pair_feat = torch.cat([x[src], x[dst]], dim=1)
        weights = self.mlp(pair_feat).squeeze(-1)
        return weights


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, use_dynamic_edge=False):
        super(GraphConvolution, self).__init__()
        self.conv_1 = GCNConv(in_channels, out_channels)
        self.bn_1 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(p=DROP_OUT_RATE)

        self.use_dynamic_edge = use_dynamic_edge
        if use_dynamic_edge:
            self.edge_predictor = EdgeWeightPredictor(in_channels)

    def forward(self, x, edge_index):
        if self.use_dynamic_edge:
            edge_weight = self.edge_predictor(x, edge_index)
        else:
            edge_weight = None

        x = self.conv_1(x, edge_index, edge_weight=edge_weight)
        x = F.gelu(self.bn_1(x))
        x = self.dropout(x)
        return x


class Expression_Path(nn.Module):
    def __init__(self, num_AU, num_of_fa_feat, num_facial_expressions, device):
        super(Expression_Path, self).__init__()

        self.gcn1_au = GraphConvolution(num_AU, int(64 * SCALE))
        self.att1_au = GGA(int(64 * SCALE))
        self.gcn2_au = GraphConvolution(int(64 * SCALE), int(128 * SCALE))
        self.att2_au = GGA(int(128 * SCALE))
        self.gcn3_au = GraphConvolution(int(128 * SCALE), int(256 * SCALE))
        self.att3_au = GGA(int(256 * SCALE))

        self.gcn1_fa_dev = GraphConvolution(10, int(64 * SCALE))
        self.att1_fa_dev = GGA(int(64 * SCALE))
        self.gcn2_fa_dev = GraphConvolution(int(64 * SCALE), int(128 * SCALE))
        self.att2_fa_dev = GGA(int(128 * SCALE))
        self.gcn3_fa_dev = GraphConvolution(int(128 * SCALE), int(256 * SCALE))
        self.att3_fa_dev = GGA(int(256 * SCALE))

        self.gcn1_fa_lr = GraphConvolution(14, int(64 * SCALE))
        self.att1_fa_lr = GGA(int(64 * SCALE))
        self.gcn2_fa_lr = GraphConvolution(int(64 * SCALE), int(128 * SCALE))
        self.att2_fa_lr = GGA(int(128 * SCALE))
        self.gcn3_fa_lr = GraphConvolution(int(128 * SCALE), int(256 * SCALE))
        self.att3_fa_lr = GGA(int(256 * SCALE))

        self.edge_list_expression = EXPRESSION_GRAPH.edges

        self.score_fcn = Projector_exp(int((256 + 128 + 64) * 8 * SCALE), int(256 * SCALE))
        self.side_fcn = Projector_exp(int((256 + 128 + 64) * 8 * SCALE), int(256 * SCALE))

        self.dropout = nn.Dropout(p=DROP_OUT_RATE)

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
        x_au_1 = self.att1_au(x_au.view(batch_size, x_au.shape[1], -1, 1)).view(-1, x_au.shape[1])
        x_au_2 = self.gcn2_au(x_au_1, edge_index_expression)
        x_au_2 = self.att2_au(x_au_2.view(batch_size, x_au_2.shape[1], -1, 1)).view(-1, x_au_2.shape[1])
        x_au_3 = self.gcn3_au(x_au_2, edge_index_expression)
        x_au_3 = self.att3_au(x_au_3.view(batch_size, x_au_3.shape[1], -1, 1)).view(-1, x_au_3.shape[1])

        x_fa_dev = self.gcn1_fa_dev(x_fa_dev, edge_index_expression)
        x_fa_dev_1 = self.att1_fa_dev(x_fa_dev.view(batch_size, x_fa_dev.shape[1], -1, 1)).view(-1, x_fa_dev.shape[1])
        x_fa_dev_2 = self.gcn2_fa_dev(x_fa_dev_1, edge_index_expression)
        x_fa_dev_2 = self.att2_fa_dev(x_fa_dev_2.view(batch_size, x_fa_dev_2.shape[1], -1, 1)).view(-1,
                                                                                                    x_fa_dev_2.shape[1])
        x_fa_dev_3 = self.gcn3_fa_dev(x_fa_dev_2, edge_index_expression)
        x_fa_dev_3 = self.att3_fa_dev(x_fa_dev_3.view(batch_size, x_fa_dev_3.shape[1], -1, 1)).view(-1,
                                                                                                    x_fa_dev_3.shape[1])

        x_fa_lr = self.gcn1_fa_lr(x_fa_lr, edge_index_expression)
        x_fa_lr_1 = self.att1_fa_lr(x_fa_lr.view(batch_size, x_fa_lr.shape[1], -1, 1)).view(-1, x_fa_lr.shape[1])
        x_fa_lr_2 = self.gcn2_fa_lr(x_fa_lr_1, edge_index_expression)
        x_fa_lr_2 = self.att2_fa_lr(x_fa_lr_2.view(batch_size, x_fa_lr_2.shape[1], -1, 1)).view(-1, x_fa_lr_2.shape[1])
        x_fa_lr_3 = self.gcn3_fa_lr(x_fa_lr_2, edge_index_expression)
        x_fa_lr_3 = self.att3_fa_lr(x_fa_lr_3.view(batch_size, x_fa_lr_3.shape[1], -1, 1)).view(-1, x_fa_lr_3.shape[1])

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

        score_output = self.score_fcn(x_au, x_fa_dev)
        side_output = self.side_fcn(x_fa_lr, x_fa_dev)

        return score_output, side_output


class AU_Path(nn.Module):
    def __init__(self, num_AU, num_of_fa_feat, num_facial_expressions, device):
        super(AU_Path, self).__init__()

        self.gcn1_au = GraphConvolution(num_facial_expressions, int(64 * SCALE))
        self.att1_au = GGA(int(64 * SCALE))
        self.gcn2_au = GraphConvolution(int(64 * SCALE), int(128 * SCALE))
        self.att2_au = GGA(int(128 * SCALE))
        self.gcn3_au = GraphConvolution(int(128 * SCALE), int(256 * SCALE))
        self.att3_au = GGA(int(256 * SCALE))

        self.edge_list_au = ACTION_UNIT_GRAPH.edges

        self.score_fcn = Projector(int(17 * (256 + 128 + 64) * SCALE), int(256 * SCALE))

        self.dropout = nn.Dropout(p=DROP_OUT_RATE)

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
        x_au_1 = self.att1_au(x_au.view(batch_size, x_au.shape[1], -1, 1)).view(-1,
                                                                                x_au.shape[1])
        x_au_2 = self.gcn2_au(x_au_1, edge_index_au)
        x_au_2 = self.att2_au(x_au_2.view(batch_size, x_au_2.shape[1], -1, 1)).view(-1, x_au_2.shape[1])
        x_au_3 = self.gcn3_au(x_au_2, edge_index_au)
        x_au_3 = self.att3_au(x_au_3.view(batch_size, x_au_3.shape[1], -1, 1)).view(-1, x_au_3.shape[1])

        x_au_1 = x_au_1.reshape(batch_size, -1)
        x_au_2 = x_au_2.reshape(batch_size, -1)
        x_au_3 = x_au_3.reshape(batch_size, -1)
        x_au = torch.cat((x_au_1, x_au_2, x_au_3), dim=1)
        score_output = self.score_fcn(x_au)

        return score_output


class FA_Path(nn.Module):
    def __init__(self, num_AU, num_of_fa_feat, num_facial_expressions, device):
        super(FA_Path, self).__init__()

        self.gcn1_fa = GraphConvolution(num_facial_expressions, int(64 * SCALE))
        self.att1_fa = GGA(int(64 * SCALE))
        self.gcn2_fa = GraphConvolution(int(64 * SCALE), int(128 * SCALE))
        self.att2_fa = GGA(int(128 * SCALE))
        self.gcn3_fa = GraphConvolution(int(128 * SCALE), int(256 * SCALE))
        self.att3_fa = GGA(int(256 * SCALE))

        self.edge_list_fa = FACIAL_ASYMMETRY_GRAPH.edges

        self.side_fcn = Projector(int(24 * (256 + 128 + 64) * SCALE), int(256 * SCALE))

        self.dropout = nn.Dropout(p=DROP_OUT_RATE)

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
        x_fa_1 = self.att1_fa(x_fa.view(batch_size, x_fa.shape[1], -1, 1)).view(-1, x_fa.shape[1])
        x_fa_2 = self.gcn2_fa(x_fa_1, edge_index_fa)
        x_fa_2 = self.att2_fa(x_fa_2.view(batch_size, x_fa_2.shape[1], -1, 1)).view(-1, x_fa_2.shape[1])
        x_fa_3 = self.gcn3_fa(x_fa_2, edge_index_fa)
        x_fa_3 = self.att3_fa(x_fa_3.view(batch_size, x_fa_3.shape[1], -1, 1)).view(-1, x_fa_3.shape[1])

        x_fa_1 = x_fa_1.reshape(batch_size, -1)
        x_fa_2 = x_fa_2.reshape(batch_size, -1)
        x_fa_3 = x_fa_3.reshape(batch_size, -1)
        x_fa = torch.cat((x_fa_1, x_fa_2, x_fa_3), dim=1)
        side_output = self.side_fcn(x_fa)
        return side_output


class HGNN(nn.Module):
    def __init__(self, num_AU, num_of_fa_feat, num_facial_expressions, device):
        super(HGNN, self).__init__()

        self.expression_path = Expression_Path(num_AU, num_of_fa_feat, num_facial_expressions, device)
        self.au_path = AU_Path(num_AU, num_of_fa_feat, num_facial_expressions, device)
        self.fa_path = FA_Path(num_AU, num_of_fa_feat, num_facial_expressions, device)

        self.weightedpred_score = Predictor()
        self.weightedpred_side = Predictor()

        self.dropout = nn.Dropout(p=DROP_OUT_RATE)

        self.num_AU = num_AU
        self.num_of_fa_feat = num_of_fa_feat
        self.num_facial_expressions = num_facial_expressions
        self.device = device

    def forward(self, x):
        x_au, x_fa = torch.split(x, [self.num_AU, x.size(1) - self.num_AU], dim=1)
        x_fa_dev, x_fa_lr = torch.split(x_fa, [10, 14], dim=1)

        batch_size = x_au.size(0) // self.num_facial_expressions

        score_expression, side_expression = self.expression_path(x_au, x_fa_dev, x_fa_lr, batch_size)
        score_au = self.au_path(x_au, batch_size)
        side_fa = self.fa_path(x_fa, batch_size)

        score = self.weightedpred_score(score_expression, score_au)
        side = self.weightedpred_side(side_expression, side_fa)

        combined_output = torch.stack([score, side], dim=1)
        return combined_output
