from __future__ import annotations

import os

import hydra
import torch
from rich import print
from sklearn.metrics import r2_score
from tqdm import tqdm

from gnn import AmazonMyGraph, Gnn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(cfg: dict):
    print(os.getcwd())
    graph_file = cfg["graph"]
    graph = AmazonMyGraph(graph_file)

    x, scores = Gnn.x_from_graph(graph)
    edge_index = Gnn.edge_index_from_graph(graph)
    x = x.float().to(DEVICE)
    edge_index = edge_index.int().to(DEVICE)
    scores = scores.float().to(DEVICE)
    assert len(scores) == len(x), [len(scores), len(x)]
    is_review = scores >= 0

    gnn = Gnn(x.shape[1], linear=cfg["linear"]).float().to(DEVICE)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=cfg["learning-rate"])
    loss_fn = torch.nn.MSELoss()

    for epoch in tqdm(range(cfg["epochs"])):
        out = gnn(x, edge_index)
        loss = loss_fn(out.pooler[is_review].squeeze(), scores[is_review])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            r2_score(
                y_pred=out.pooler.detach()[is_review].cpu().numpy(),
                y_true=scores.clone()[is_review].cpu().numpy(),
            )
        )

    gnn.save(cfg["save-path"])


if __name__ == "__main__":
    main()
