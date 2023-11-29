import shutil
import pandas as pd
import os

data = pd.read_csv("../train.csv")

# move image to the label directory, if the directory does not exist, create it
for i in range(len(data)):
    image_name = data["image_name"][i]
    label = data["label"][i]
    if not os.path.exists(f"../data/train/{label}"):
        os.mkdir(f"../data/train/{label}")
    shutil.move(f"../data/train/{image_name}", f"../data/train/{label}/{image_name}")