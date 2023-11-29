import re, gc, sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import torchvision.models as models


def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.cuda.manual_seed(worker_id)
    os.environ['PYTHONHASHSEED'] = str(worker_id)

def seed_everything(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

#データセットモジュールを定義します
class Custom_Dataset(torch.utils.data.Dataset):
    #事前に定義を行う箇所
    def __init__(self, df, transform):
        self.df = df
        self.image_paths = df['image_name']
        self.labels = df['label']
        self.transform= transform

    #画像データの枚数を数える箇所
    def __len__(self):
        return len(self.df)

    #画像を読み込み前処理を行う箇所
    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = self.labels[index]
        #画像読み込み
        image = cv2.imread(f"data/train/{image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #画像の前処理を行う関数です
        image = self.transform(image)

        #モデルに学習されるにはデータをtensor型に変換する必要があります。
        label = torch.tensor(label, dtype=torch.long)

        return image, label



data = pd.read_csv("train.csv")

# hyperparameters
EPOCHS = 4
# learning rate
LR = 1e-5
# batch size
TRAIN_BATCH_SIZE, VALID_BATCH_SIZE = 8, 8
# image size
size = (256, 256)
# seed
seeds = 777

image_transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train, valid, _, _ = train_test_split(data, data["label"], random_state=777)
train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)

train_data = Custom_Dataset(train, image_transform)
train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                            worker_init_fn=worker_init_fn(seeds), pin_memory=True)

valid_data = Custom_Dataset(valid, image_transform)
valid_loader = DataLoader(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=False, pin_memory=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

seed_everything(seeds)
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss().to(device)

print(train.shape, valid.shape)

best_score = -np.inf
for epoch in range(1, 1+EPOCHS):
  #モデルの訓練フェーズ
  model.train()
  train_loss = []
  for process, (images, targets) in enumerate(train_loader):
    images = images.to(device)
    targets = targets.to(device)
    outputs = model(images)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    train_loss.append(loss.item())
  print(f'EPOCH{epoch} train_loss: {np.mean(train_loss)}')

  #モデル評価のフェーズ
  model.eval()
  softmax = nn.Softmax()
  loss_list = []
  predict_list, targets_list = [], []
  for process, (images, targets) in enumerate(valid_loader):
    images = images.to(device)
    targets = targets.to(device)
    with torch.no_grad():
      outputs = model(images)
      predict = outputs.softmax(dim=1)
      loss = criterion(outputs, targets)

    predict = predict.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    loss_list.append(loss.item())
    predict_list.append(predict)
    targets_list.append(targets)

  predict_list, targets_list = np.concatenate(predict_list, axis=0), np.concatenate(targets_list)
  predict_list_proba = predict_list.copy()[:, 1]
  predict_list = predict_list.argmax(axis=1)

  score = accuracy_score(predict_list, targets_list)
  auc_score = roc_auc_score(targets_list, predict_list_proba)

  print('EPOCH{} loss {:.4f} | ACC {:.4f} | AUC {:.4f}'. format(epoch, np.mean(loss_list), score, auc_score))

  if auc_score >= best_score:
    best_score = auc_score
    torch.save(model.state_dict(), f"BEST_Model.pth")
    valid["predict"] = predict_list_proba

fpr, tpr, thresholds = roc_curve(valid["label"], valid["predict"])

plt.plot(fpr, tpr, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.show()

print("BEST AUC")
print(roc_auc_score(valid["label"], valid["predict"]))





