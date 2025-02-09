import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch.nn import Linear, ReLU, BatchNorm1d
import torch.nn as nn
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_dense_adj

class SuperpixelsMPNN(MessagePassing):
    def __init__(self, in_channels, hidden_channels):
        super(SuperpixelsMPNN, self).__init__(aggr='sum')
        
        self.mlp1 = nn.Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            BatchNorm1d(hidden_channels)
        )
        
        self.mlp2 = nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            ReLU(),
            BatchNorm1d(hidden_channels)
        )

    def forward(self, x, edge_index):
        x = self.mlp1(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return torch.cat([x_i, x_j], dim=1)

    def update(self, aggr_out):
        return self.mlp2(aggr_out)

class LaplacianPE(BaseTransform):
    def __init__(self, k=8):
        self.k = k

    def __call__(self, data):
        edge_index, edge_weight = get_laplacian(
            data.edge_index, normalization='sym', 
            num_nodes=data.num_nodes
        )
        
        adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
        
        try:
            eigvals, eigvecs = torch.linalg.eigh(adj)
        except RuntimeError:
            eigvecs = torch.zeros((data.num_nodes, self.k))
            print(f"Warning: Eigendecomposition failed for a graph. Using zeros.")
            data.pos_enc = eigvecs
            return data

        idx = torch.argsort(eigvals)
        eigvecs = eigvecs[:, idx]
        pos_enc = eigvecs[:, 1:self.k+1]
        data.pos_enc = pos_enc

        return data

class Net(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=10):
        super(Net, self).__init__()
        
        hidden_channels = 64
        
        self.conv1 = SuperpixelsMPNN(num_features, hidden_channels)
        self.conv2 = SuperpixelsMPNN(hidden_channels, hidden_channels)
        self.conv3 = SuperpixelsMPNN(hidden_channels, hidden_channels)
        
        self.classifier = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            BatchNorm1d(hidden_channels),
            Linear(hidden_channels, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)

class NetWithPE(torch.nn.Module):
    def __init__(self, num_features=1, pos_enc_dim=8, num_classes=10):
        super(NetWithPE, self).__init__()
        
        hidden_channels = 64
        input_dim = num_features + pos_enc_dim
        
        self.conv1 = SuperpixelsMPNN(input_dim, hidden_channels)
        self.conv2 = SuperpixelsMPNN(hidden_channels, hidden_channels)
        self.conv3 = SuperpixelsMPNN(hidden_channels, hidden_channels)
        
        self.classifier = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            BatchNorm1d(hidden_channels),
            Linear(hidden_channels, num_classes)
        )

    def forward(self, data):
        x = torch.cat([data.x, data.pos_enc], dim=-1)
        edge_index, batch = data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)