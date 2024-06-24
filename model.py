import torch
import numpy as np
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geo_nn
from gnn import GIN, GAT


class BasicCountNet(nn.Module):
    def __init__(self, input_feat_dim, query_hidden_dim, data_hidden_dim, out_dim, pooling_method='sumpool', share_net=False):
        super(BasicCountNet, self).__init__()
        self.pool_method = pooling_method
        if not share_net:
            self.query_GNN = GIN(input_feat_dim, query_hidden_dim, out_dim)
            self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        else:
            self.query_GNN = self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(2* out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim//2),
            nn.ReLU(),
            nn.Linear(out_dim//2, 1),
            nn.ReLU()
        )

    def pool_operation(self, x):
        # build batch
        num_nodes = x.size(0)
        batch = torch.from_numpy(np.zeros(num_nodes)).type(torch.LongTensor)
        if self.pool_method == 'sumpool':
            return geo_nn.global_add_pool(x, batch)
        elif self.pool_method == 'meanpool':
            return geo_nn.global_mean_pool(x, batch)
        elif self.pool_method == 'maxpool':
            return geo_nn.global_max_pool(x, batch)
        else:
            raise NotImplementedError

    def forward(self, query_in_feat, data_in_feat, query_edge_list, data_edge_list, query2data_edge_list=None):
        query_x = self.query_GNN(query_in_feat, query_edge_list)
        data_x = self.data_GNN(data_in_feat, data_edge_list)
        query_x = self.pool_operation(query_x)
        data_x = self.pool_operation(data_x)
        # do we need unsqueeze?
        out_feat = torch.cat((query_x, data_x), dim=1)
        pred = self.linear_layers(out_feat)
        return pred


class AttentiveCountNet(nn.Module):
    def __init__(self, input_feat_dim, query_hidden_dim, data_hidden_dim, out_dim, pooling_method='sumpool', share_net=False):
        super(AttentiveCountNet, self).__init__()
        self.pool_method = pooling_method
        if not share_net:
            self.query_GNN = GIN(input_feat_dim, query_hidden_dim, out_dim)
            self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        else:
            self.query_GNN = self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        self.attention_layer = GAT(input_feat_dim, out_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(4* out_dim, 2*out_dim),
            nn.Linear(2* out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim//2),
            nn.ReLU(),
            nn.Linear(out_dim//2, 1),
            nn.ReLU()
        )

    def pool_operation(self, x):
        # build batch
        num_nodes = x.size(0)
        batch = torch.from_numpy(np.zeros(num_nodes)).type(torch.LongTensor)
        if self.pool_method == 'sumpool':
            return geo_nn.global_add_pool(x, batch)
        elif self.pool_method == 'meanpool':
            return geo_nn.global_mean_pool(x, batch)
        elif self.pool_method == 'maxpool':
            return geo_nn.global_max_pool(x, batch)
        else:
            raise NotImplementedError

    def forward(self, query_in_feat, data_in_feat, query_edge_list, data_edge_list, query2data_edge_list):
        query_x = self.query_GNN(query_in_feat, query_edge_list)
        data_x = self.data_GNN(data_in_feat, data_edge_list)
        num_query_vertices = query_x.shape[0]
        query2data_in_feat = torch.cat((query_in_feat, data_in_feat), dim=0)
        query2data_x = self.attention_layer(query2data_in_feat, query2data_edge_list)
        query_x_with_data = query2data_x[:num_query_vertices, :]
        data_x_with_query = query2data_x[num_query_vertices:, :]
        out_query_x = torch.cat((query_x, query_x_with_data), dim=1)
        out_data_x = torch.cat((data_x, data_x_with_query),dim=1)
        # do we need unsqueeze?
        query_x = self.pool_operation(out_query_x)
        data_x = self.pool_operation(out_data_x)
        out_feat = torch.cat((query_x, data_x), dim=1)
        # note that we can change the scale function.
        pred = self.linear_layers(out_feat)
        return pred, out_query_x, out_data_x


class WasserstainDiscriminator(nn.Module):
    def __init__(self, hidden_dim, hidden_dim2=512):
        super(WasserstainDiscriminator, self).__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(hidden_dim2, 1)
        )
    
    def forward(self, input_x):
        return self.linear_layers(input_x)


class QErrorLoss:
    def __init__(self):
        pass

    def __call__(self, input_card, true_card):
        q_error = torch.max(torch.cat(((torch.max(torch.cat((input_card.unsqueeze(0), torch.tensor([1]))))/torch.max(torch.cat((true_card.unsqueeze(0), torch.tensor([1]))))).unsqueeze(0), (torch.max(torch.cat((true_card.unsqueeze(0), torch.tensor([1]))))/torch.max(torch.cat((input_card.unsqueeze(0), torch.tensor([1]))))).unsqueeze(0))))
        return q_error


class QErrorLikeLoss:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def __call__(self, input_card, true_card):
        loss = torch.max(torch.cat(((true_card/(input_card+self.epsilon)).unsqueeze(0), (input_card/(true_card+self.epsilon)).unsqueeze(0))))
        return loss


class CoarsenNet(nn.Module):
    def __init__(self, input_feat_dim, query_hidden_dim, data_hidden_dim, out_dim, pooling_method='sumpool', share_net=False):
        super(CoarsenNet, self).__init__()
        self.pool_method = pooling_method
        if not share_net:
            self.query_GNN = GIN(input_feat_dim, query_hidden_dim, out_dim)
            self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        else:
            self.query_GNN = self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(2* out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim//2),
            nn.ReLU(),
            nn.Linear(out_dim//2, 1),
            nn.ReLU()
        )
