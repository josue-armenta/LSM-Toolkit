import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

class PreprocessLayer(nn.Module):
    def __init__(self,
                 emg_rate: int = 500,
                 target_rate: int = 100,
                 use_wavelet: bool = False,
                 wavelet: str = "bior3.9",
                 level: int = 4):
        super().__init__()
        self.emg_rate = emg_rate
        self.target_rate = target_rate
        self.use_wavelet = use_wavelet
        self.wavelet = wavelet
        self.level = level

    def _wavelet_denoise(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        wav = pywt.Wavelet(self.wavelet)
        max_level = pywt.dwt_max_level(x.shape[-1], wav.dec_len)
        lvl = min(self.level, max_level)
        # Decompose
        coeffs = pywt.wavedec(x.cpu().numpy(), wav, level=lvl, axis=-1)
        # Noise estimate
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-8
        uthresh = sigma * np.sqrt(2 * np.log(x.shape[-1]))
        # Soft-threshold
        coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
        # Reconstruct
        denoised = pywt.waverec(coeffs, wav, axis=-1)
        # Crop to original length
        denoised = denoised[..., :x.shape[-1]]
        return torch.from_numpy(denoised).to(x.dtype).to(x.device)

    def _interp(self, x: torch.Tensor, new_t: int) -> torch.Tensor:
        # Linear interpolation along time axis
        return F.interpolate(x, size=new_t, mode="linear", align_corners=False)

    def _align(self,*tensors: torch.Tensor):
        # encuentra la longitud mínima
        T = min(t.size(-1) for t in tensors)
        # recorta todos al mismo T
        return [t[..., :T] for t in tensors]

    def forward(self,
                emg: torch.Tensor,
                acc: torch.Tensor,
                gyro: torch.Tensor,
                euler: torch.Tensor,
                quat: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            emg: (B, T, 8)
            acc: (B, T, 3)
            gyro: (B, T, 3)
            euler: (B, T, 3)
            quat: (B, T, 4)
        Returns:
            Tensor of shape (B, 21, new_T)
        """
        # Transpose to (B, C, T)
        emg   = emg.transpose(1, 2)
        acc   = acc.transpose(1, 2)
        gyro  = gyro.transpose(1, 2)
        euler = euler.transpose(1, 2)
        quat  = quat.transpose(1, 2)

        # Denoise
        if self.use_wavelet:
            emg   = self._wavelet_denoise(emg)
            acc   = self._wavelet_denoise(acc)
            gyro  = self._wavelet_denoise(gyro)
            # euler and quat usually low-noise; skip or denoise if needed

        # Resample EMG
        orig_t = emg.shape[-1]
        new_t  = int(orig_t * self.target_rate / self.emg_rate)
        emg_ds = self._interp(emg, new_t)

        # Alineacion
        emg, acc, gyro, euler, quat = self._align(emg, acc, gyro, euler, quat)

        # Combine IMU and resample
        imu    = torch.cat((acc, gyro, euler, quat), dim=1)
        imu_rs = self._interp(imu, new_t)

        # Concatenate EMG + IMU = 8 + (3+3+3+4) = 21 channels
        out = torch.cat((emg_ds, imu_rs), dim=1)
        assert out.shape[1] == 21, f"Preprocess output has {out.shape[1]} channels, expected 21"
        return out
