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
    edge_index, scores = Gnn.edge_index_from_graph(graph)
    edge_index = edge_index.to(DEVICE)
    scores = scores.to(DEVICE)
    assert len(scores) == len(x), [len(scores), len(x)]
    is_review = scores >= 0

    gnn = Gnn(x.shape[1]).to(DEVICE)
    print(gnn)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=cfg["learning_rate"])
    loss_fn = torch.nn.MSELoss()

    for epoch in tqdm(range(cfg["epochs"])):
        out = gnn(x, edge_index)
        loss.backward()
        optimizer.zero_grad()
        loss = loss_fn(out.pooler[is_review], scores[is_review])
        optimizer.step()
        print(out.raw.shape, out.pooler.shape)
        # input()


if __name__ == "__main__":
    main()
