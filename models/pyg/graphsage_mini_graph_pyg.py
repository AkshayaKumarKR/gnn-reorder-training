import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from tqdm import tqdm


class SAGE_PYG(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, hidden_channels=128):
        super(SAGE_PYG, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels, 'mean'))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, 'mean'))
            for _ in range(1, self.num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, 'mean'))
            self.convs.append(SAGEConv(hidden_channels, out_channels, 'mean'))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, test_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in test_loader:
                edge_index, _, size = adj.to("cpu")
                total_edges += edge_index.size(1)
                x = x_all[n_id].to("cpu")
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all
