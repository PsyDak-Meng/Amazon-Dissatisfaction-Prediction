import json
import os
import random

with open("Fashion_data.json") as f:
    data = json.load(f)

ids = list(data.keys())

shuffled = random.shuffle(ids)

LABELLED_FILE = "labelled-helpfulness.json"
if os.path.exists(LABELLED_FILE):
    with open(LABELLED_FILE) as f:
        result = json.load(f)
else:
    result = {}


for idx, dat_id in enumerate(ids, start=1):
    os.system("clear")

    print(
        f"{idx})\n\nDescription:\n"
        + data[dat_id]["description"]
        + "\n\nReview:\n"
        + data[dat_id]["review"]
        + "\n\nHelpful? y/n (q to quit)>",
        end="",
    )

    while True:
        key = input().lower()
        if key == "y":
            result[dat_id] = "helpful"
            break
        elif key == "n":
            result[dat_id] = "muda"
            break
        elif key == "q":
            with open(LABELLED_FILE, "w+") as f:
                json.dump(result, f, indent=4)
            raise SystemExit
        else:
            print("Please answer with one of y/n (q to quit) >", end="")
