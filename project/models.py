import torch.nn as nn
from torchvision import models


class ResNetDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.main = models.resnet18()
        self.regressor = nn.Sequential(
            nn.Linear(1000, 4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.main(x)
        x = self.regressor(x)
        
        return x


class ImageDecoder(nn.Module):
    def __init__(self, num_vocab, embedding_dim=128) -> None:
        super().__init__()

        self.embedding = nn.Embedding(num_vocab, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=1000, num_layers=1, batch_first=True)

        self.predictor = nn.Sequential(
            nn.Linear(1000, num_vocab),
            nn.Softmax(dim=1),
        )


    def forward(self, x_txt, x_img):
        txt_emb = self.embedding(x_txt)
        h0 = x_img
        c0 = x_img

        out, _ = self.lstm(txt_emb, (h0, c0))

        out = self.predictor(out)
        return out
