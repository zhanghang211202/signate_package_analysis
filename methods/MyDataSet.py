import torch
import cv2

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.image_paths = df['image_name']
        self.labels = df['label']
        self.transform= transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = cv2.imread(f"data/train/train/{image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)

        return image, label