from vit_L import ViTLightningModule

model = ViTLightningModule().load_from_checkpoint('./lightning_logs/lightning_logs/version_3/checkpoints/epoch=47-step=36576.ckpt')
from pytorch_lightning import Trainer
trainer=Trainer()
predictions = trainer.predict(model, test_ds)