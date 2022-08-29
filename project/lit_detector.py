import pytorch_lightning as pl
import torch as th
from torchvision.ops import generalized_box_iou_loss

from project.models import ResNetDetector


class LicensePlateDetectorModule(pl.LightningModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = ResNetDetector()
        self.loss_fn = generalized_box_iou_loss


    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=0.00003)

    
    def training_step(self, batch, batch_idx):
        x, bbox = batch
        preds = self(x)
        
        loss = self.loss_fn(preds, bbox).mean()

        self.log('train/loss', loss)
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, bbox = batch
        preds = self(x)

        loss = self.loss_fn(preds, bbox).mean()
        
        self.log("val/loss", loss)

        return loss

    
    def test_step(self, batch, batch_idx):
        x, bbox = batch
        preds = self(x)

        loss = self.loss_fn(preds, bbox).mean()
        
        self.log("test/loss", loss)

        return loss
