from __future__ import annotations
import os
import numpy as np
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

    x, scores, helpfulness, goodness = Gnn.x_from_graph(graph)
    edge_index = Gnn.edge_index_from_graph(graph)

    x = x.float().to(DEVICE)
    edge_index = edge_index.int().to(DEVICE)
    scores = scores.float().to(DEVICE)
    helpfulness = helpfulness.float().to(DEVICE)
    goodness = goodness.float().to(DEVICE)

    assert len(scores) == len(x), [len(scores), len(x)]
    assert len(helpfulness) == len(x), [len(helpfulness), len(x)]
    assert len(goodness) == len(x), [len(goodness), len(x)]
    is_review = scores >= 0
    is_helpful = ~helpfulness.isnan()
    is_good = ~goodness.isnan()

    helpfulness = torch.isclose(helpfulness, torch.ones_like(helpfulness)).float()
    goodness = torch.isclose(goodness, torch.ones_like(goodness)).float()

    gnn = Gnn(x.shape[1], linear=cfg["linear"]).float().to(DEVICE)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=cfg["learning-rate"])
    loss_mse = torch.nn.MSELoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    results = np.zeros((4,cfg("epochs"))) # epoch,loss,mse,r2
    for epoch in tqdm(range(cfg["epochs"])):
        out = gnn(x, edge_index)
        loss = 0
        results[0,epoch] = epoch
        loss += loss_mse(out.pooler_score[is_review].squeeze(), scores[is_review])
        results[2,epoch] = loss
        loss += loss_bce(
            out.pooler_helpful[is_helpful].squeeze(), helpfulness[is_helpful]
        )
        loss += loss_bce(out.pooler_good[is_good].squeeze(), goodness[is_good])
        results[1,epoch] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        R2 = r2_score(
                y_pred=out.pooler_score.detach()[is_review].cpu().numpy(),
                y_true=scores.clone()[is_review].cpu().numpy(),
            )
        results[3,epoch] = R2
        print(R2)

    gnn.save(cfg["save-path"])
    with open('results.npy','w') as f:
        np.save(f,results)

if __name__ == "__main__":
    main()
