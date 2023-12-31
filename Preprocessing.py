import gzip
import json
import math
import os
from argparse import ArgumentParser
from pathlib import Path

import parse
import requests
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class AmazonDataset(Dataset):
    def __init__(self, review_fname: str, product_fname: str, img_loaded: bool):
        super().__init__()
        self._data = {}
        self._product = {}

        ############## Read from product dataset ############
        def download_img(urls, asin, img_loaded: bool):
            if img_loaded:
                return True
            else:
                # Opening the image and displaying it (to confirm its presence)
                imgfiles = [f for f in os.listdir(Path(Path.cwd(), "Images"))]
                # Skip if img in file already
                if f"{asin}.png" in imgfiles:
                    return True
                else:
                    count = 0
                    for url in urls:
                        try:
                            img = Image.open(requests.get(url, stream=True).raw)
                            self._imgURLs.append(url)
                            path = Path(Path.cwd(), "Images", f"{asin}.png")
                            img.save(path)
                            break
                        except:
                            count += 1
                    if count == len(urls):
                        return False
                    else:
                        return True

        f_product = gzip.open(product_fname, "rb")
        lines = f_product.readlines()
        # self._model = SentenceTransformer("all-MiniLM-L6-v2")
        for i in tqdm(range(len(lines))):
            line = lines[i]
            dp = json.loads(line)

            if "title" in dp.keys() and "imageURL" in dp.keys():
                product = dp["title"]
                img = dp["imageURL"]
                asin = dp["asin"]
            else:
                continue

            if "description" in dp.keys():
                description = dp["description"]
                description = " ".join(description)
            else:
                continue

            if not download_img(img, asin, img_loaded):
                continue

            meta = ""
            for ft in ["brand", "feature"]:
                if ft in dp.keys():
                    meta += " ".join(dp[ft])
            self._product[asin] = {
                "product": product,
                "description": description,
                "meta": meta,
            }
        f_product.close()
        print(f"{len(self._product)} products")

        def sentiment_scores(sentence):
            # Create a SentimentIntensityAnalyzer object.
            sid_obj = SentimentIntensityAnalyzer()

            # polarity_scores method of SentimentIntensityAnalyzer
            # object gives a sentiment dictionary.
            # which contains pos, neg, neu, and compound scores.
            sentiment_dict = sid_obj.polarity_scores(sentence)

            """ print("Overall sentiment dictionary is : ", sentiment_dict)
            print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
            print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
            print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
        
            print("Sentence Overall Rated As", end = " ")
        
            # decide sentiment as positive, negative and neutral
            if sentiment_dict['compound'] >= 0.05 :
                return "Positive"
            elif sentiment_dict['compound'] <= - 0.05 :
                return "Negative"
            else :
                return "Neutral"
                """
            return sentiment_dict["neg"]

        ############ Read from review dataset ##############
        def merge_dict(a, b):
            a.update(b)
            return a

        f_review = gzip.open(review_fname, "rb")
        # Initialize self._data
        lines = f_review.readlines()
        reviewID = 0

        for i in tqdm(range(len(lines))):

            line = lines[i]
            dp = json.loads(line.decode("utf-8"))
            asin = dp["asin"]
            if asin in self._product.keys():
                if "reviewText" not in dp.keys():
                    continue

                review = (
                    dp["reviewText"] + dp["summary"]
                    if "summary" in dp.keys()
                    else dp["reviewText"]
                )
                score = float(dp["overall"])
                verified = dp["verified"]
                reviewerID = dp["reviewerID"]
                reviewername = (
                    dp["reviewerName"] if "revieweName" in dp.keys() else "Anonymous"
                )
                _raw_rank = dp.get("rank")
                rank = (
                    parse.parse("{}in{}", _raw_rank)[0]
                    if _raw_rank is not None
                    else str(len(lines))
                )
                rank = int(rank.replace(",", "").replace(" ", ""))
                
                reviewID += 1
                self._data[str(reviewID)] = {
                    "reviewerID" : reviewerID,
                    "reviewerName": reviewername,
                    "verified": verified,
                    "asin": asin,
                    "score": score,
                    "review": review,
                    "log_rank": -math.log(rank),
                }
                self._data[str(reviewID)] = merge_dict(
                    self._data[str(reviewID)], self._product[asin]
                )

            # TODO: Add matching criteria for dissatisfactory #

        f_review.close()

        # Show example
        def print_dict_example(dict, idx):
            reviewerID = list(dict.keys())[idx]
            print(f"Datapoint Example:\nreviewerID : {reviewerID}")
            for key in list(dict[reviewerID].keys()):
                print(f"{key} : {dict[reviewerID][key]}")

        print_dict_example(self._data, 0)
        print(f"Total reviews {len(self._data)}")

    def __getitem__(self, index: int):
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", default="Dataset/Amazon Review")
    flags = vars(parser.parse_args())

    path = Path(flags["path"])
    Fashion = AmazonDataset(
        review_fname=path / "AMAZON_FASHION.json.gz",
        product_fname=path / "meta_AMAZON_FASHION.json.gz",
        img_loaded=True,
    )

    with open("Fashion_data.json", "w") as f:
        json.dump(Fashion._data, f, indent=4)
