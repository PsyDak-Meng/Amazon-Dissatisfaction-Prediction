from __future__ import annotations

import abc
from typing import NamedTuple, Protocol, Sequence

from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import GCNConv


class ReviewRecord(NamedTuple):
    review_id: str
    product_id: str
    user_id: str


class AmazonGraph(Protocol):
    @abc.abstractmethod
    def reviews_by_users(self, user_id: str) -> Sequence[ReviewRecord]:
        ...

    @abc.abstractmethod
    def reviews_by_products(self, product_id: str) -> Sequence[ReviewRecord]:
        ...

    @abc.abstractmethod
    def review(self, review_id: str) -> dict:
        ...

    @abc.abstractmethod
    def product(self, product_id: str) -> dict:
        ...

    @abc.abstractmethod
    def user(self, user_id: str) -> dict:
        ...

    @abc.abstractmethod
    def user_embedding(self, user_id: str) -> NDArray:
        ...

    @abc.abstractmethod
    def product_embedding(self, product_id: str) -> NDArray:
        ...

    @abc.abstractmethod
    def review_embedding(self, review_id: str) -> NDArray:
        ...


class Gnn(Module):
    def __init__(self, in_channels: int, out_channels: int, K: int) -> None:
        self.gcn = GCNConv(in_channels=in_channels, out_channels=in_channels)
        self.out = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.K = K

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for _ in range(self.K):
            x = self.gcn(x, edge_index)
        x = self.out(x)
        return x
