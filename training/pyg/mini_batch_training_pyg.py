import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import graphlearn_torch as glt
from tqdm import tqdm


def mini_batch_training_pyg(model, config, g, timer=None):
    train_idx = list(torch.nonzero(g.train_mask).squeeze())

    # Initialise graphlearn_torch Dataset
    glt_dataset = glt.data.Dataset()

    # Initialise graphlearn_torch Graph
    glt_dataset.init_graph(
        edge_index=g.edge_index,
        graph_mode='CPU',
        directed=False
    )

    # Initialise graphlearn_torch Node Features
    glt_dataset.init_node_features(
        node_feature_data=g.x,
        # sort_func=glt.data.sort_by_in_degree,
        # split_ratio=0.2,
        device_group_list=[glt.data.DeviceGroup(0, [0])],
        with_gpu=False
    )

    # Initialise graphlearn_torch Labels
    glt_dataset.init_node_labels(node_label_data=g.y)

    # graphlearn_torch NeighborLoader
    train_loader = glt.loader.NeighborLoader(glt_dataset,
                                             config["neighbors_per_layer"],  # [25, 10, 5]
                                             train_idx,
                                             batch_size=config["batch_size"],
                                             shuffle=True,
                                             drop_last=True,
                                             device=torch.device('cpu'),
                                             as_pyg_v1=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, config['num_epochs'] + 1):
        pbar = tqdm(total=len(train_idx))
        pbar.set_description(f'Epoch {epoch}')
        timer.start("Epoch")
        epoch_start = time.time()
        model.train()
        total_loss = total_correct = step = 0
        glt_dataset.node_labels = glt_dataset.node_labels.to("cpu")
        # Data Loading Start
        for batch_size, n_id, adjs in train_loader:
            adjs = [adj.to("cpu") for adj in adjs]
            # Data Loading End
            # Learning Start
            optimizer.zero_grad()
            # Forward pass through the model using batched data
            out = model(glt_dataset.node_features[n_id], adjs)
            loss = F.nll_loss(out, glt_dataset.node_labels[n_id[:batch_size]])
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(glt_dataset.node_labels[n_id[:batch_size]]).sum())
            # print(out.argmax(dim=-1), glt_dataset.node_labels[n_id[:batch_size]])
            step += 1  # Increment the batch counter
            # Learning End
            pbar.update(batch_size)

        epoch_end = time.time()
        timer.stop("Epoch")
        pbar.close()
        avg_loss = total_loss / step
        approx_acc = total_correct / len(train_idx)
        print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Approx Train Accuracy: {approx_acc:.4f}, '
              f'Epoch Time: {epoch_end - epoch_start:.2f}')
