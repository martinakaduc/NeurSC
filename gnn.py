import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geo_nn


class GIN(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim, out_dim, train_eps=True):
        super(GIN, self).__init__()
        # we can change the sequential nn
        nn_module_for_gin_1 = nn.Sequential(
            nn.Linear(input_feat_dim, hidden_dim),
            nn.ReLU()
        )
        nn_module_for_gin_2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )
        self.GIN_layer_1 = geo_nn.GINConv(nn_module_for_gin_1, train_eps=train_eps)
        self.GIN_layer_2 = geo_nn.GINConv(nn_module_for_gin_2, train_eps=train_eps)

    def forward(self, in_feat, edge_list):
        x = self.GIN_layer_1(in_feat, edge_list)
        x = self.GIN_layer_2(x, edge_list)

        return x


class GAT(nn.Module):
    def __init__(self, input_feat_dim, out_dim, train_eps=True):
        super(GAT, self).__init__()
        # we can change the sequential nn
        self.GAT_layer = geo_nn.GATConv(input_feat_dim, out_dim, add_self_loops=False)

    def forward(self, in_feat, edge_list):
        x = self.GAT_layer(in_feat, edge_list)
        # x = self.GIN_layer_2(x, edge_list)

        return x