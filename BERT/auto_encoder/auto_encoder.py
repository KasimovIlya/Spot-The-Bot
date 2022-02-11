import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.training = True

        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=128, out_features=hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=input_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.training:
            x = self.decoder(x)
        return x

    def train(self, mode=True):
        self.training = mode
        for param in self.parameters():
            param.requires_grad = mode
        return self

    def eval(self):
        model = self.train(False)
        self.training = True
        return model

    def test(self):
        return self.train(False)
