import torch.nn.functional as F
import torch.nn
from dgl.nn.pytorch.conv import GraphConv
from tqdm import tqdm
import dgl


class GCN_DGL(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, hidden_channels=128):
        super(GCN_DGL, self).__init__()

        self.num_layers = num_layers
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.layers = torch.nn.ModuleList()

        if num_layers == 1:
            self.layers.append(
                GraphConv(in_channels, out_channels))
        else:
            self.layers.append(GraphConv(in_channels, hidden_channels))
            for _ in range(1, self.num_layers - 1):
                self.layers.append(GraphConv(hidden_channels, hidden_channels))
            self.layers.append(GraphConv(hidden_channels, out_channels))
        self.dropout = torch.nn.Dropout(0.5)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.

            # Create a DGL graph from edge_index
            src, dst = edge_index
            g = dgl.graph((src, dst), num_nodes=x.size(0)).to(torch.device("cpu"))

            # Add self-loops to the graph
            g = dgl.add_self_loop(g)

            x = self.layers[i](g, x)
            if i != self.num_layers - 1:
                x = F.relu(x)
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
                    x = F.relu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all
