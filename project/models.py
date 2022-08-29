import torch as th
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
    def __init__(self, num_vocab, embedding_dim=128, hidden_size=1000) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_vocab = num_vocab
        self.embedding = nn.Embedding(num_vocab, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, num_vocab),
        )


    def forward(self, x_txt, x_img):
        seq_len = x_txt.shape[1]

        txt_emb = self.embedding(x_txt)
        x_img = x_img.unsqueeze(0)
        h0 = x_img
        c0 = x_img

        
        out, (ht, ct) = self.lstm(txt_emb, (h0, c0))

        out = self.predictor(out)
        return out.view(seq_len, self.num_vocab) # remove batch dimension
