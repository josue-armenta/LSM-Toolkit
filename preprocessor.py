# preprocessor.py
import torch
import torch.nn.functional as F
from torch import nn

class PreprocessLayer(nn.Module):
    def __init__(self, emg_rate: float = 250, target_rate: float = 100):
        """
        emg_rate: frecuencia de muestreo original de EMG
        target_rate: frecuencia deseada (por ejemplo, IMU_RATE)
        """
        super().__init__()
        self.emg_rate = emg_rate
        self.target_rate = target_rate

    def _interp(self, x: torch.Tensor, new_T: int) -> torch.Tensor:
        return F.interpolate(x, size=new_T, mode="linear", align_corners=False)

    def forward(self, emg: torch.Tensor, acc: torch.Tensor,
                gyro: torch.Tensor, euler: torch.Tensor,
                quat: torch.Tensor) -> torch.Tensor:
        # Emular exactamente la lógica original
        emg   = emg.transpose(1, 2)
        imu   = torch.cat((acc, gyro, euler, quat), dim=2).transpose(1, 2)
        new_T = int(round(emg.shape[2] * self.target_rate / self.emg_rate))
        emg_ds = self._interp(emg, new_T)
        imu_rs = self._interp(imu, new_T)
        return torch.cat((emg_ds, imu_rs), dim=1)  # B,21,T
