from __future__ import annotations

import abc
import itertools
import json
from collections import OrderedDict
from typing import Literal, NamedTuple, Protocol, Sequence

import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.nn import LeakyReLU, Linear, Module
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512
BATCH = 2048


class ReviewRecord(NamedTuple):
    review_id: str
    product_id: str
    user_id: str


class AmazonMyGraph:
    def __init__(self, file_path):
        with open(file_path) as f:
            self._data = json.load(f)

        self._reviews_by_users = OrderedDict({})
        self._reviews_by_products = OrderedDict({})
        self._review_ids = [i for i in range(1, len(list(self._data.keys())) + 1)]
        self._reviews = OrderedDict({})  # reviewID -> review
        self._user_ids = []  # user_id
        self._users = OrderedDict({})  # user_id -> userName
        self._products = OrderedDict({})  # product_id -> product

        self._sentence = SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE)

        for i in self._review_ids:
            self._user_ids.append(self._data[str(i)]["reviewerID"])
            self._users[self._data[str(i)]["reviewerID"]] = self._data[str(i)][
                "reviewerName"
            ]
            self._reviews[str(i)] = self._data[str(i)]["review"]
            self._products[self._data[str(i)]["asin"]] = self._data[str(i)]["product"]
        self._user_ids = list(set(self._user_ids))
        self._products_reversedict = {
            value: key for key, value in self._products.items()
        }

        # reviews_by_users
        for user_id in list(self._users.keys()):
            self._reviews_by_users[user_id] = []
        for i in self._review_ids:
            self._reviews_by_users[self._data[str(i)]["reviewerID"]].append(
                self._data[str(i)]["review"]
            )

        # reviews_by_products
        for product_id in list(self._products.keys()):
            self._reviews_by_products[product_id] = []
        for i in self._review_ids:
            self._reviews_by_products[
                self._products_reversedict[self._data[str(i)]["product"]]
            ].append(self._data[str(i)]["review"])

    def sentence_transform(self, input: list[str]):
        return self._sentence.encode(input)

    def reviews_by_users(self, user_id: str) -> Sequence[ReviewRecord]:
        return self._reviews_by_users[user_id]

    def reviews_by_products(self, product_id: str) -> Sequence[ReviewRecord]:
        return self._reviews_by_products[product_id]

    # Unsure what returned dict should contain, returning only respective strings rn
    def review(self, review_id: str) -> dict:
        return self._reviews[review_id]

    def product(self, product_id: str) -> dict:
        return self._products[product_id]

    def user(self, user_id: str) -> dict:
        return self._users[user_id]

    def user_embedding(self, user_id: str) -> NDArray:
        return self.BERT_embed(self.user(user_id))

    def product_embedding(self, product_id: str) -> NDArray:
        return self.BERT_embed(self.product(product_id))

    def review_embedding(self, review_id: str) -> NDArray:
        return self.BERT_embed(self.review(review_id))

    def get_ids(
        self, types: Literal["user", "product", "review", None] = None, k: int = -1
    ):
        if k >= 0:
            print(f"Top {k} ids:")
            print("User ids:", dict(itertools.islice(self._users.items(), k)))
            print("Product ids:", dict(itertools.islice(self._products.items(), k)))
            print("Review ids:", dict(itertools.islice(self._reviews.items(), k)))

        # gets corresponding values to embed by _ids lists/_products dict values orders; KEEEP ORDER !!!
        user_fn = lambda: self._user_ids
        # gets userNames from user_id
        product_fn = lambda: list(
            self._products.keys()
        )  # gets product names from products dict values
        review_fn = lambda: [
            str(i) for i in self._review_ids
        ]  # gets reviews from review_id
        if types == None:
            return user_fn() + product_fn() + review_fn()
        elif types == "user":
            return user_fn()
        elif types == "product":
            return product_fn()
        elif types == "review":
            return review_fn()
        else:
            raise ValueError

    def edges(self):
        edges = []
        # get index order for user, product and review; KEEP ORDER !!!
        UPR_cat = (
            []
            + self._user_ids
            + [self._products[product] for product in list(self._products.keys())]
            + [self._reviews[review] for review in self._review_ids]
        )
        UPR = {upr: i for i, upr in enumerate(UPR_cat)}

        # create edge matrix
        for dp in self._data:
            # one user for one review
            edges.append((UPR[dp["review"]], UPR[dp["reviewerID"]]))
            # one product for one review
            edges.append(UPR[dp["review"]], UPR[dp["product"]])
        return np.array(edges)


class Gnn(Module):
    def __init__(self, dims: int) -> None:
        super().__init__()

        self.gnn1 = GCNConv(in_channels=dims, out_channels=dims)
        self.gnn2 = GCNConv(in_channels=2 * dims, out_channels=dims)
        self.relu = LeakyReLU()

        self.output = Linear(in_features=dims, out_features=1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        assert edge_index.shape[0] == 2

        y = self.gnn1(x, edge_index)
        y = torch.cat([x, y], dim=-1)
        y = self.relu(y)

        z = self.gnn2(y, edge_index)
        z = self.relu(z)

        a = self.output(z)
        return a

    @staticmethod
    def x_from_graph(graph: AmazonMyGraph):
        user_ids = graph.get_ids(types="user")
        product_ids = graph.get_ids(types="product")
        review_ids = graph.get_ids(types="review")

        # embed by _ids orders
        user_data = [graph.user(id) for id in user_ids]
        product_data = [graph.product(id) for id in product_ids]
        review_data = [graph.review(id) for id in review_ids]

        user_embeddings = [
            graph.sentence_transform(user_data[idx : idx + BATCH])
            for idx in tqdm(range(0, len(user_data), BATCH))
        ]
        product_embeddings = [
            graph.sentence_transform(product_data[idx : idx + BATCH])
            for idx in tqdm(range(0, len(product_data), BATCH))
        ]
        review_embeddings = [
            graph.sentence_transform(review_data[idx : idx + BATCH])
            for idx in tqdm(range(0, len(review_data), BATCH))
        ]

        user_emb_dim = user_embeddings.shape[1]
        product_emb_dim = product_embeddings.shape[1]
        review_emb_dim = review_embeddings.shape[1]

        user_embeddings = np.concatenate(
            [
                user_embeddings,
                np.zeros((1, product_emb_dim + review_emb_dim)),
            ],
            axis=1,
        )
        product_embeddings = np.concatenate(
            [
                np.zeros(1, user_emb_dim),
                product_embeddings,
                np.zeros((1, review_emb_dim)),
            ],
            axis=1,
        )
        review_embeddings = np.concatenate(
            [
                np.zeros(1, user_emb_dim + product_emb_dim),
                review_embeddings,
            ],
            axis=1,
        )

        x = np.concatenate([user_embeddings, product_embeddings, review_embeddings])
        return torch.from_numpy(x).float()

    @staticmethod
    def edge_index_from_graph(graph: AmazonMyGraph):
        edge_index = []

        id_idx = {id: idx for idx, id in enumerate(graph.get_ids())}
        edges = graph.edges()
        for x, y in edges:
            x_idx = id_idx[x]
            y_idx = id_idx[y]

            edge_index.append((x_idx, y_idx))

        return torch.tensor(edge_index).int()


if __name__ == "__main__":
    MyGraph = AmazonMyGraph("Fashion_data.json")
    MyGraph.get_ids(k=3)

    logger.info("creating x")
    x = Gnn.x_from_graph(MyGraph)
    logger.info("creating edge")
    edge_index = Gnn.edge_index_from_graph(MyGraph)
    logger.info("creating gnn")
    gnn = Gnn(dims=x.shape[1])
    logger.info("gnn forward")
    out = gnn(x=x, edge_index=edge_index)
    print(out.shape, x.shape, edge_index.shape)
