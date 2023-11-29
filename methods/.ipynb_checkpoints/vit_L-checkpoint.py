from datasets import load_dataset
from transformers import ViTImageProcessor
# load dataset
train_ds, test_ds = load_dataset('imagefolder', data_dir='../data', split=['train', 'test'])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.3)
train_ds = splits['train']
val_ds = splits['test']
id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}
print(id2label)







