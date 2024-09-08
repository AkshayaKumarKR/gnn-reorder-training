import os
from ogb.nodeproppred import DglNodePropPredDataset

def save_edgelist(graph, graph_name):
    dir = f"input/{graph_name}"
    if not os.path.exists(dir):
        os.system(f"mkdir -p {dir}")
        src, dst = graph.edges()
        print("Create edgelist in memeory", graph_name)
        edge_list = list(zip(src.tolist(), dst.tolist()))
        num_edges = len(edge_list)
        c = 0
        with open(f"{dir}/{graph_name}", 'w') as f:
            print("Start writing out edgelist")
            for e in edge_list:
                c += 1
                if (num_edges % 1000000 == 0):
                    print(f"{c} edges are written out")
                f.write(f"{e[0]}\t{e[1]}\n")

        if num_edges != c:
            print("it seems  like that all edges have been written out", c, "  ", num_edges)
    else:
        print(f"Graph is already available {graph_name}")


# data_products = DglNodePropPredDataset(name='ogbn-products')
# graph_products, _ = data_products[0]
# save_edgelist(graph=graph_products, graph_name="ogbn-products")

data_arxiv = DglNodePropPredDataset(name='ogbn-arxiv')
graph_arxiv, _ = data_arxiv[0]
save_edgelist(graph=graph_arxiv, graph_name="ogbn-arxiv")
