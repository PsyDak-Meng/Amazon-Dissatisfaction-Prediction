from __future__ import annotations

import os

import hydra
import torch
from rich import print
from tqdm import tqdm

from gnn import AmazonMyGraph, Gnn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(cfg: dict):
    print(os.getcwd())
    graph_file = cfg["graph"]
    graph = AmazonMyGraph(graph_file)

    x = Gnn.x_from_graph(graph).to(DEVICE)
    edge_index = Gnn.edge_index_from_graph(graph).to(DEVICE)

    gnn = Gnn(x.shape[1]).to(DEVICE)
    print(gnn)

    for epoch in tqdm(range(cfg["epochs"])):
        out = gnn(x, edge_index)
        print(out.raw.shape, out.pooler.shape)
        # input()


if __name__ == "__main__":
    main()
