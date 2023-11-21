import gzip
import json
from typing import Dict

import networkx as nx
import numpy as np


class DSNConstants:
    REVIEWER_ID = "reviewerID"
    REVIEWER_NAME = "reviewerName"
    REVIEW_RATING = "overall"
    REVIEW_TEXT = "reviewText"
    REVIEW_SUMMARY = "summary"
    REVIEW_TIMESTAMP = "unixReviewTime"
    PRODUCT_ID = "asin"
    PRODUCT_TITLE = "title"
    PRODUCT_DESCRIPTION = "description"
    PRODUCT_IMAGE_URLs = "imageURL"
    PRODUCT_HIGH_RES_IMAGE_URLs = "imageURLHighRes"


"""
    Key --> 'reviewerID'
    value --> list of dictionaries with the following structure:
                (key, value) --> ('rating', ???),
                (key, value) --> ('reviewerName', ???),
                (key, value) --> ('reviewText', ???),
                (key, value) --> ('summary', ???),
                (key, value) --> ('unixReviewTime', ???)
"""


def Load(file_path: str):
    lines = []
    with gzip.open(file_path) as f:
        for line in f:
            lines.append(line.rstrip())
    return lines


def Reviews(data_fashion_file: str):
    fashion_reviews = {}
    for line in Load(data_fashion_file):
        d = json.loads(line)

        reviewerID = None
        reviewerName = None
        reviewRating = None
        reviewText = None
        reviewSummary = None
        reviewTimeStamp = None
        productId = None

        if DSNConstants.REVIEWER_ID in d:
            reviewerID = d[DSNConstants.REVIEWER_ID]
        else:
            continue

        if DSNConstants.PRODUCT_ID in d:
            productId = d[DSNConstants.PRODUCT_ID]
        else:
            continue

        if DSNConstants.REVIEW_RATING in d:
            reviewRating = d[DSNConstants.REVIEW_RATING]
        else:
            continue

        if DSNConstants.REVIEWER_NAME in d:
            reviewerName = d[DSNConstants.REVIEWER_NAME]

        if DSNConstants.REVIEW_TEXT in d:
            reviewText = d[DSNConstants.REVIEW_TEXT]

        if DSNConstants.REVIEW_SUMMARY in d:
            reviewSummary = d[DSNConstants.REVIEW_SUMMARY]

        if DSNConstants.REVIEW_TIMESTAMP in d:
            reviewTimeStamp = d[DSNConstants.REVIEW_TIMESTAMP]

        if productId not in fashion_reviews:
            fashion_reviews[productId] = []

        review = {
            DSNConstants.REVIEWER_NAME: reviewerName,
            DSNConstants.REVIEW_RATING: reviewRating,
            DSNConstants.REVIEW_TEXT: reviewText,
            DSNConstants.REVIEW_SUMMARY: reviewSummary,
            DSNConstants.REVIEW_TIMESTAMP: reviewTimeStamp,
            DSNConstants.REVIEWER_ID: reviewerID,
        }

        fashion_reviews[productId].append(review)

    return fashion_reviews


"""
    Key --> 'asin' (unique product id)
    value --> dictionary with the following structure:
                (key, value) --> ('title', ???),
                (key, value) --> ('imageURL', ???),
                (key, value) --> ('description', ???),
"""


def Products(data_fashion_meta_file: str):
    fashion_products = {}
    for line in Load(data_fashion_meta_file):
        d = json.loads(line)

        productId = None
        productTitle = None
        productDescription = None
        productImageURLs = None
        productHighResImageURLs = None

        if DSNConstants.PRODUCT_ID in d:
            productId = d[DSNConstants.PRODUCT_ID]
        else:
            continue

        if DSNConstants.PRODUCT_TITLE in d:
            productTitle = d[DSNConstants.PRODUCT_TITLE]
        else:
            continue

        if DSNConstants.PRODUCT_DESCRIPTION in d:
            productDescription = d[DSNConstants.PRODUCT_DESCRIPTION]
        else:
            continue

        if DSNConstants.PRODUCT_IMAGE_URLs in d:
            productImageURLs = d[DSNConstants.PRODUCT_IMAGE_URLs]

        if DSNConstants.PRODUCT_HIGH_RES_IMAGE_URLs in d:
            productHighResImageURLs = d[DSNConstants.PRODUCT_HIGH_RES_IMAGE_URLs]

        if productImageURLs == None and productHighResImageURLs == None:
            continue

        fashion_products[productId] = {
            DSNConstants.PRODUCT_TITLE: productTitle,
            DSNConstants.PRODUCT_DESCRIPTION: productDescription,
            DSNConstants.PRODUCT_IMAGE_URLs: productImageURLs,
            DSNConstants.PRODUCT_HIGH_RES_IMAGE_URLs: productHighResImageURLs,
        }
    return fashion_products


def GenerateGraph(fashion_products: Dict, fashion_reviews: Dict):
    G_text = nx.Graph()

    for productId in fashion_products.keys():
        description = ""
        for d in fashion_products[productId][DSNConstants.PRODUCT_DESCRIPTION]:
            if d != None:
                description += d
        G_text.add_node(description)
        for review in fashion_reviews[productId]:
            r = ""
            if review[DSNConstants.REVIEW_SUMMARY] != None:
                r += review[DSNConstants.REVIEW_SUMMARY]
            if review[DSNConstants.REVIEW_TEXT] != None:
                r += review[DSNConstants.REVIEW_TEXT]
            G_text.add_node(r)
            G_text.add_edge(description, r)

    return G_text


if __name__ == "__main__":
    FASHION_REVIEW = "AMAZON_FASHION.json.gz"
    FASHION_META = "meta_AMAZON_FASHION.json.gz"

    fashion_products = Products(FASHION_META)
    fashion_reviews = Reviews(FASHION_REVIEW)

    print("lenghts", len(fashion_products), len(fashion_reviews))
    sizes = set()
    for productId in fashion_products.keys():
        sizes.add(len(fashion_products[productId]["description"]))
        print(fashion_products[productId]["description"])
        break

    print(sizes)

    G_text = GenerateGraph(fashion_products, fashion_reviews)
    print(len(G_text.nodes()))
    print(len(G_text.edges()))
