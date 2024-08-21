from models.pyg.graphsage_mini_graph_pyg import SAGE_PYG
from models.pyg.gcn_mini_graph_pyg import GCN_PYG
from models.pyg.gat_mini_graph_pyg import GAT_PYG
from models.dgl.graphsage_mini_graph_dgl import SAGE_DGL
from models.dgl.gcn_mini_graph_dgl import GCN_DGL
from models.dgl.gat_mini_graph_dgl import GAT_DGL


class Models:
    def get_models():
        return {
            "dgl": {
                    "GRAPHSAGE": SAGE_DGL,
                    "GCN": GCN_DGL,
                    "GAT": GAT_DGL,
                },

            "pyg": {
                    "GRAPHSAGE": SAGE_PYG,
                    "GCN": GCN_PYG,
                    "GAT": GAT_PYG,
                }
            }
