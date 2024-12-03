import torch.nn as nn

class TransformerForRegression(nn.Module):
    def __init__(self, input_dim):
        super(TransformerForRegression, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=256),
            num_layers=2
        )
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids):
        encoded_output = self.encoder(input_ids.unsqueeze(1)).squeeze(1)
        output = self.regressor(encoded_output)
        return output