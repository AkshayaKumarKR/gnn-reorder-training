# GNN Training of Reordered Graphs using AliGraph

In this implemented sytem using AliGraph, we study effectiveness of graph reordering in GNN training by varying GNN hyperparameters.



## Run the experiments

Follow below steps:

&nbsp;1. Clone the repository:
```bash
  git clone https://github.com/AkshayaKumarKR/gnn-reorder-training.git
```
&nbsp;2. Create virtual environment: 
```bash
  cd gnn-reorder-training/
  python -m venv gnnvenv
  source gnnvenv/bin/activate
```
&nbsp;3. Install dependencies: 
```bash
  bash install_dependencies.sh
```
&nbsp;4. Download Graphs:
```bash
  python utils/download_graphs.py
```
&nbsp;5. Pre-process Graphs:
```bash
  bash preprocess_graphs.sh
```
&nbsp;6. Reorder Graphs:
```bash
  bash reorder_graphs.sh
```
&nbsp;7. Create PyG Graphs:
```bash
  bash bash create_pyg_graphs.sh
```

## Run the experiments
To run the experiments, we must set parameters accordingly. The
following parameters are used in the experiment:

– graph_name: The name of the graph, e.g., ogbn-arxiv. 

– model: The GNN model to train, e.g., GCN or GRAPHSAGE. 

– reordering_strategy: The reordering strategy to use, e.g. rand-0, rand-1, rand-2, metis-16, metis-128, metis-1024, metis-8192, metis-65536, rabbit, DegSort, dfs, gorder, Hub-Cluster, HubSort, ldg, rcm, slashburn, minla, bfs.

– system: The system to use for training. Use "dgl" for Deep Graph Library (DGL) or "pyg" for PyTorch Geometric (PyG).

– batch_size: Since we are using mini-batch training. Specify the number of target vertices to sample in a single batch. The default batch size is 1024.

– num_epochs: Sets the number of training epochs. Defaults to 8.

– num_layers: Indicates the number of layers in the neural network. Defaults to 2.

– neighbors_per_layer: Defines the number of neighbors to sample per layer for NeighborSampler (e.g., 15 10 5 for 3 layers of sampling).

– num_features: This defines the number of features for each input node. Defaults to 16.

– hidden_dim: This sets the size of the hidden layer dimension. Defaults to 16.
– path_to_result_metrics: Specify the file path where the result metrics will be stored (e.g., "/gnn-reorder-training/experiments/1.json").

#### Example: 
```bash
training/run_training.py -graph_name ogbn-arxiv -model GRAPHSAGE -
reordering_strategy rand-0 -system dgl -neighbors_per_layer 15 10 5 -
batch_size 2048 -num_epochs 10 -num_features 16 -num_layers 3 -hidden_dim
16 -path_to_result_metrics experiments/1.json
```
We also try to calculate cache misses during the experiment to understand the number of times the CPU could not find the requested data in the CPU cache, which is a small, fast memory storage located closer to the CPU than the main memory (RAM). When a cache miss occurs, the CPU must fetch the required data from a lower level of memory (like the main memory), which is slower and results in increased execution time. To get cache misses, we need to run the experiment with sudo permissions to log all the information. The below example shows how to generate a cache misses log for an experiment. Replace <Path to Python virtual environment> with the virtual environment path used in the experiment.
```bash
sudo sh -c ’echo 3 > /proc/sys/vm/drop_caches’
sudo perf stat -e branch-instructions,branch-misses,cache-misses,cache-
references,cpu-cycles,instructions,stalled-cycles-backend,stalled-cycles-
frontend,alignment-faults,bpf-output,context-switches,cpu-clock,cpu-
migrations,dummy,emulation-faults,major-faults,minor-faults,page-faults,
task-clock,duration_time,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-
prefetches,L1-icache-load-misses,L1-icache-loads <Path to Python virtual
environment>
training/run_training.py -graph_name ogbn-arxiv -model GRAPHSAGE -
reordering_strategy rand-0 -system dgl -neighbors_per_layer 15 10 5 -
batch_size 2048 -num_epochs 10 -num_features 16 -num_layers 3 -hidden_dim
16 -path_to_result_metrics experiments/1.json > experiments/1.cache.json
2>&1
```
