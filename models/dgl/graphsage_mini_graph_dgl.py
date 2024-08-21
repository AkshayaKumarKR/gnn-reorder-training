import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl


class SAGE_DGL(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, hidden_channels=128):
        super(SAGE_DGL, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(dglnn.SAGEConv(in_channels, out_channels, 'mean'))
        else:
            self.convs.append(dglnn.SAGEConv(in_channels, hidden_channels, 'mean'))
            for _ in range(1, self.num_layers - 1):
                self.convs.append(dglnn.SAGEConv(hidden_channels, hidden_channels, 'mean'))
            self.convs.append(dglnn.SAGEConv(hidden_channels, out_channels, 'mean'))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            # Extract the target nodes' features
            x_target = x[:size[1]]

            # Split the edge_index into source and destination nodes
            src, dst = edge_index

            # Calculate the correct number of nodes
            num_nodes = max(src.max().item(), dst.max().item()) + 1

            # Create the DGLGraph object using the separated source and destination nodes
            graph = dgl.graph((src, dst), num_nodes=num_nodes)

            # Pass the graph and node features through the SAGEConv layer
            x = self.convs[i](graph, x)

            # After applying the convolution, the output for target nodes must be selected
            x = x[:size[1]]

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)