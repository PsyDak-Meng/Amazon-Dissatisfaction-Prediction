from __future__ import annotations
import json
import abc
from typing import NamedTuple, Protocol, Sequence
from transformers import AutoTokenizer, AutoModel
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module
import itertools
#from torch_geometric.nn import GCNConv


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


class AmazonMyGraph(AmazonGraph):
    def __init__(self,file_path):
        with open(file_path) as f:
            self._data = json.load(f)
         
        self._reviews_by_users = {}
        self._reviews_by_products = {}
        self._review_ids = [i for i in range(1,len(list(self._data.keys()))+1)]
        self._reviews = {} # reviewID -> review
        self._user_ids = [] # user_id
        self._users = {} # user_id -> userName
        self._products = {} # product_id -> product

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        

        for i in self._review_ids:
            self._user_ids.append(self._data[str(i)]['reviewerID'])
            self._users[self._data[str(i)]['reviewerID']] = self._data[str(i)]['reviewerName']
            self._reviews[str(i)] = self._data[str(i)]['review']
            self._products[self._data[str(i)]['asin']] = self._data[str(i)]['product']
        self._user_ids = list(set(self._user_ids))
        self._products_reversedict = {value:key for key,value in self._products.items()}

        # reviews_by_users
        for user_id in list(self._users.keys()):
            self._reviews_by_users[user_id] = []
        for i in self._review_ids:
            self._reviews_by_users[self._data[str(i)]['reviewerID']].append(self._data[str(i)]['review'])  

        # reviews_by_products
        for product_id in list(self._products.keys()):
            self._reviews_by_products[product_id] = []
        for i in self._review_ids:   
            self._reviews_by_products[self._products_reversedict[self._data[str(i)]['product']]].append(self._data[str(i)]['review'])

        
    def BERT_embed(self,input:str):
        inputs = self.tokenizer(input, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs

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
        return self.BERT_embed(self._products[product_id])

    def review_embedding(self, review_id: str) -> NDArray:
        return self.BERT_embed(self.review(review_id))
    
    # TEST
    def get_ids(self,k:int):
        print(f"Top {k} ids:")
        print('User ids:', dict(itertools.islice(self._users.items(), k)))
        print('Product ids:', dict(itertools.islice(self._products.items(), k)))
        print('Review ids:', dict(itertools.islice(self._reviews.items(), k)))



""" class Gnn(Module):
    def __init__(self, in_channels: int, out_channels: int, K: int) -> None:
        self.gcn = GCNConv(in_channels=in_channels, out_channels=in_channels)
        self.out = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.K = K

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for _ in range(self.K):
            x = self.gcn(x, edge_index)
        x = self.out(x)
        return x """


if __name__ == "__main__":
    MyGraph = AmazonMyGraph('Fashion_data.json')
    MyGraph.get_ids(3)