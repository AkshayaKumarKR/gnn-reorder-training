import torch
import time
import dgl
import torch.nn.functional as F
import graphlearn_torch as glt

from tqdm import tqdm


def mini_batch_training_dgl(model, config, g, timer=None):

    train_idx = torch.nonzero(g.ndata["train_mask"]).squeeze().to(dtype=torch.int64).to(torch.device("cpu"))

    # Initialise graphlearn_torch Dataset
    glt_dataset = glt.data.Dataset()

    # Initialise graphlearn_torch Graph
    glt_dataset.init_graph(
        edge_index=torch.stack(g.edges()),
        graph_mode='CPU',
        directed=False
    )

    # Initialise graphlearn_torch Node Features
    glt_dataset.init_node_features(
        node_feature_data=g.ndata['features'],
        sort_func=glt.data.sort_by_in_degree,
        # split_ratio=0.2,
        device_group_list=[glt.data.DeviceGroup(0, [0])],
        with_gpu=False
    )

    # Initialise graphlearn_torch Labels
    glt_dataset.init_node_labels(node_label_data=g.ndata['labels'].to(torch.device("cpu")))

    # graphlearn_torch NeighborLoader
    train_loader = glt.loader.NeighborLoader(glt_dataset,
                                             config["neighbors_per_layer"],  # [25, 10, 5]
                                             train_idx,
                                             batch_size=config["batch_size"],
                                             shuffle=True,
                                             drop_last=False,
                                             device=torch.device('cpu'),
                                             as_pyg_v1=True)

    model = model.to("cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, config['num_epochs'] + 1):
        pbar = tqdm(total=train_idx.size(0))
        pbar.set_description(f'Epoch {epoch}')
        epoch_start = time.time()
        timer.start("Epoch")
        model.train()
        total_loss = total_correct = step = 0
        glt_dataset.node_labels = glt_dataset.node_labels.to("cpu")
        timer.start("Dataloading")
        for i, (batch_size, n_id, adjs) in enumerate(train_loader):
            adjs = [adj.to("cpu") for adj in adjs]
            timer.stop("Dataloading")
            timer.start("Learning")
            x = glt_dataset.node_features[n_id]
            y = glt_dataset.node_labels[n_id[:batch_size]]

            optimizer.zero_grad()
            y_hat = model(x, adjs)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            total_correct += int(y_hat.argmax(dim=-1).eq(y).sum())
            step += 1
            timer.stop("Learning")
            pbar.update(batch_size)

        timer.stop("Epoch")
        pbar.close()
        avg_loss = total_loss / step
        approx_acc = total_correct / len(train_idx)
        print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Approx Train Accuracy: {approx_acc:.4f}')
        print(total_correct, len(train_idx))
