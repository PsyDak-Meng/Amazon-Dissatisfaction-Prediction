# DSN
Data Science for Social Network Project on Amazon Dissatisfaction Detection using Graph Network &amp; Textual &amp; Image data.

## Preprocessing
Run

```
python Preprocessing.py
```

The result would be saved in a `Fashion_data.json` file. This file is required for training.

### Statistics
- 148516 Reviews, 9723 Products with reviews, 438 Products with images
- asin: product Id<br>

**Datapoint Example:<br>**
- **key**<br>
reviewerID : A1BB77SEBQT8VX<br>
- **value**<br>
reviewerName : Anonymous<br>
verified : True<br>
asin : B00007GDFV<br>
score : 3.0<br>
review : mother - in - law wanted it as a present for her sister. she liked it and said it would work.bought as a present<br>
product : Buxton Heiress Pik-Me-Up Framed Case<br>
description : Authentic crunch leather with rich floral embossed logo heiress pik-me-up framed case features a large pocket, outside slip pocket and outside zipper pocket.       <br>
meta : B u x t o nLeather Imported synthetic lining Flap closure Authentic Crunch Leather Rich Floral Embossed Logo Goldtone Hardware Large Snap Pocket You can return this item for any reason and get a full refund: no shipping charges. The item must be returned in new and unused condition. Read the full returns policy Go to Your Orders to start the return Print the return shipping label Ship it! Product Dimensions:5 x 6 x 2 inches Shipping Weight:0.8 ounces (View shipping rates and policies)

## Training
Simply run 
```
python train.py epoch=???
```

To perform training on the obtained graph in `Fashion_data.json`. Our own results is trained over 1000 epochs.

If a GPU is detected, it would be used (running without GPU is extrememly slow).
