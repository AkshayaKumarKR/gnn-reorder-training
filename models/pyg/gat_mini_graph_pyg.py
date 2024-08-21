import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from tqdm import tqdm


class GAT_PYG(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2, heads=8, activation=F.elu):
        super(GAT_PYG, self).__init__()

        self.num_layers = num_layers

        self.layers = torch.nn.ModuleList()
        self.activation = activation

        hidden_channels = int(hidden_channels)

        if num_layers == 1:
            self.layers.append(
                GATv2Conv(in_channels, out_channels, 1))
        else:
            self.layers.append(
                GATv2Conv(in_channels, hidden_channels // heads, heads))
            for _ in range(1, num_layers - 1):
                self.layers.append(
                    GATv2Conv(hidden_channels, hidden_channels // heads, heads))
            self.layers.append(
                GATv2Conv(hidden_channels, out_channels, 1))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.layers[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, test_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in test_loader:
                edge_index, _, size = adj.to('cpu')
                total_edges += edge_index.size(1)
                x = x_all[n_id].to('cpu')
                x_target = x[:size[1]]
                x = self.layers[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all
