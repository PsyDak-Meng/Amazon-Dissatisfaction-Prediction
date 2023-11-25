from __future__ import annotations

import os

import hydra
from rich import print
from tqdm import tqdm

from gnn import AmazonMyGraph, Gnn


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(cfg: dict):
    print(os.getcwd())
    graph_file = cfg["graph"]
    graph = AmazonMyGraph(graph_file)

    x = Gnn.x_from_graph(graph)
    edge_index = Gnn.edge_index_from_graph(graph)

    gnn = Gnn(x.shape[1])
    print(gnn)

    for epoch in tqdm(range(cfg["epochs"])):
        out = gnn(x, edge_index)
        print(out.shape)
        input()


if __name__ == "__main__":
    main()
