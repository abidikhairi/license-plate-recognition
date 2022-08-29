import pytorch_lightning as pl
import torch as th
from torchvision import models
import torch.nn.functional as F
import torchmetrics.functional as thm
from project.models import ImageDecoder
from project.utils import word2idx


class LicensePlateRecognitionModule(pl.LightningModule):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_vocab = len(word2idx)

        self.image_encoder = models.resnet18()
        self.model = ImageDecoder(num_vocab=self.num_vocab)
        self.loss_fn = F.cross_entropy


    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=0.00003)
    

    def forward(self, seq, img):
        img = self.image_encoder(img)
        return self.model(seq, img)

    
    def training_step(self, batch, batch_idx):
        image, seq = batch
        target = seq.squeeze(0)

        output = self(seq, image)
        loss = self.loss_fn(output, target)

        self.log('train/loss', loss)

        return loss


    def validation_step(self, batch, batch_idx):
        image, seq = batch
        target = seq.squeeze(0)
        
        output = self(seq, image)
        loss = self.loss_fn(output, target)
        acc = thm.accuracy(F.softmax(output, dim=1), target)
        
        self.log('valid/loss', loss)
        self.log('valid/acc', acc)

        return loss


    def test_step(self, batch, batch_idx):
        image, seq = batch
        target = seq.squeeze(0)

        output = self(seq, image)
        loss = self.loss_fn(output, target)
        acc = thm.accuracy(F.softmax(output, dim=1), target)

        self.log('test/loss', loss)
        self.log('test/acc', acc)

        return loss
