import torch.nn as nn
from preprocessor import PreprocessLayer

class SignNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        emg_rate: int = 500,
        target_rate: int = 100,
        use_wavelets: bool = False,
        conv_ch=(64, 128),
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        self.prep = PreprocessLayer(emg_rate=emg_rate, target_rate=target_rate, use_wavelet=use_wavelets)
        self.conv = nn.Sequential(
            nn.Conv1d(21, conv_ch[0], 5, padding=2), nn.BatchNorm1d(conv_ch[0]), nn.ReLU(),
            nn.Conv1d(conv_ch[0], conv_ch[1], 5, padding=2), nn.BatchNorm1d(conv_ch[1]), nn.ReLU()
        )
        self.lstm = nn.LSTM(conv_ch[1], lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.attn = nn.Linear(lstm_hidden*2, 1, bias=False)
        self.cls  = nn.Linear(lstm_hidden*2, num_classes)

    def forward(self, batch):
        emg, acc, gyro, euler, quat = batch
        x = self.prep(emg, acc, gyro, euler, quat)        # (B,21,T)
        x = self.conv(x).transpose(1, 2)                  # (B,T,C)
        h, _ = self.lstm(x)                               # (B,T,2H)
        a = (self.attn(h).softmax(dim=1))                 # (B,T,1)
        ctx = (a * h).sum(dim=1)                          # (B,2H)
        return self.cls(ctx)
