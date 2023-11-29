from datasets import load_dataset
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification

# load dataset
train_ds, test_ds = load_dataset('imagefolder', data_dir='../data', split=['train', 'test'])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.2)
train_ds = splits['train']
val_ds = splits['test']
id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label: id for id, label in id2label.items()}


processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["height"]

from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

normalize = Normalize(mean=image_mean, std=image_std)
if "height" in processor.size:
    size = (processor.size["height"], processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in processor.size:
    size = processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = processor.size.get("longest_edge")

_train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

from torch.utils.data import DataLoader
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_batch_size = 16
eval_batch_size = 16

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size)
test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=1)

import pytorch_lightning as pl
from transformers import ViTForImageClassification, AdamW
import torch.nn as nn


class ViTLightningModule(pl.LightningModule):
    def __init__(self, num_labels=2):
        super(ViTLightningModule, self).__init__()
        self.swin = AutoModelForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224',
                                                             num_labels=num_labels,
                                                             ignore_mismatched_sizes=True,
                                                             id2label=id2label,
                                                             label2id=label2id)

    def forward(self, pixel_values):
        outputs = self.swin(pixel_values=pixel_values)
        return outputs.logits

    def predict_step(self, pixel_values):
        return self(pixel_values)

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=5e-5)

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

    def test_dataloader(self):
        return test_dataloader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers

# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
# early_stop_callback = EarlyStopping(
#     monitor='val_loss',
#     min_delta=0.00,
#     patience=50,
#     verbose=False,
#     mode='min'
# )
# checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="validation_accuracy", mode="max")
# tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/")
# model = ViTLightningModule()
#
# trainer = Trainer(
#     num_nodes=1,
#     max_epochs=150,
#     callbacks=[checkpoint_callback],
#     val_check_interval=len(train_dataloader),
#     logger=tb_logger
# )
# trainer.fit(model)

model = ViTLightningModule().load_from_checkpoint('./lightning_logs/lightning_logs/version_15/checkpoints/epoch=140-step=15369.ckpt')
model.eval()

# read the test set in ../data/test/0 and predict
import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

test_dir = '../data/test/0'
test_images_name = os.listdir(test_dir)
test_images_path = [os.path.join(test_dir, image) for image in test_images_name]
test_images = [Image.open(image) for image in test_images_path]
test_images_pixel_values = [_val_transforms(image.convert("RGB")) for image in test_images]

# create test_images_dict, key is image name, value is test_images_pixel_value
test_images_dict = {}
for i in range(len(test_images_name)):
    test_images_dict[test_images_name[i]] = test_images_pixel_values[i]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# predict
pred_dict = {}

for image_name, image_pixel_values in test_images_dict.items():
    image_pixel_values = image_pixel_values.unsqueeze(0)
    image_pixel_values = image_pixel_values.to(device)
    output = model.predict_step(image_pixel_values)
    output = output.softmax(dim=1).cpu().detach().numpy()
    pred_dict[image_name] = output[0][1]

# save the prediction
import pandas as pd
# save the prediction to csv that contains two columns: image_name, label
pred_df = pd.DataFrame.from_dict(pred_dict, orient='index', columns=['label'])
pred_df.index.name = 'image_name'
pred_df.to_csv('submission_swin.csv')




















# trainer=Trainer()
# predictions = trainer.predict(model, test_dataloader)
# print(predictions)