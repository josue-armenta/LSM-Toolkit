# trainer.py — End‑to‑end trainer con CUDA por defecto
"""Entrenador end‑to‑end para gForcePro+ (Wang 2020‑like) — *CLI*

Uso mínimo
----------
```bash
python trainer.py --root data --classes 3 --epochs 40 --batch 32
```

— Carpetas esperadas: `session_*/user_*/phrase_<id>/rep_*/data.h5` (id numérico).
— Guarda el mejor modelo en `best_signnet.pt` con `num_classes` incluido.
"""
from __future__ import annotations
import argparse, glob, os, random, re
from dataclasses import dataclass
from typing import List, Tuple
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from model import SignNet

# reproducibilidad
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

###############################################################################
# 1. Dataset
###############################################################################
class GForceTrialDataset(Dataset):
    _RX = re.compile(r"[\\/]+phrase_(\d+)(?=[\\/])", re.IGNORECASE)
    def __init__(self, files: List[str]):
        if not files:
            raise ValueError("Empty H5 list")
        self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        with h5py.File(path, 'r') as f:
            emg  = torch.from_numpy(f['raw/emg'] [:,1:].astype('f'))
            acc  = torch.from_numpy(f['raw/acc'] [:,1:].astype('f'))
            gyro = torch.from_numpy(f['raw/gyro'][:,1:].astype('f'))
            eul  = torch.from_numpy(f['raw/euler'][:,1:].astype('f'))
            quat = torch.from_numpy(f['raw/quat'][:,1:].astype('f'))
        m = self._RX.search(path)
        if m is None:
            raise ValueError(f"Bad path, no 'phrase_X': {path}")
        label = int(m.group(1)) - 1
        return (emg, acc, gyro, eul, quat), torch.tensor(label)

###############################################################################
# 2. Collate
###############################################################################
_pad = lambda x, T: F.pad(x, (0,0,0, T - x.size(0))) if x.size(0) < T else x

def collate(batch):
    xs, ys = zip(*batch)
    Te = max(s[0].size(0) for s in xs)
    Ti = max(max(s[j].size(0) for j in range(1,5)) for s in xs)
    out = [[] for _ in range(5)]
    for emg, acc, gyro, eul, quat in xs:
        out[0].append(_pad(emg, Te)); out[1].append(_pad(acc, Ti))
        out[2].append(_pad(gyro, Ti)); out[3].append(_pad(eul, Ti))
        out[4].append(_pad(quat, Ti))
    return tuple(torch.stack(z) for z in out), torch.tensor(ys)

###############################################################################
# 5. Config + Entrenamiento
###############################################################################
@dataclass
class CFG:
    root: str
    classes: int
    batch: int = 16
    workers: int = 4
    epochs: int = 50
    lr: float = 1e-3
    val: float = 0.2
    patience: int = 8
    wd: float = 1e-4
    emg_rate: int = 500

def train(cfg: CFG, dev: torch.device):
    # Patrón de archivos
    patt = os.path.join(cfg.root, 'session_*/user_*/phrase_*/rep_*/data.h5')
    files = sorted(glob.glob(patt))
    if not files: raise FileNotFoundError('No se encontraron archivos H5')

    # Split
    random.shuffle(files)
    k = int((1 - cfg.val) * len(files))
    tr, va = files[:k], files[k:]

    # DataLoaders
    pin = (dev.type == 'cuda')
    dl_tr = DataLoader(
        GForceTrialDataset(tr),
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        collate_fn=collate,
        pin_memory=pin
    )
    dl_va = DataLoader(
        GForceTrialDataset(va),
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=cfg.workers,
        collate_fn=collate,
        pin_memory=pin
    )

    # Modelo y optimizador
    net = SignNet(cfg.classes, use_wavelets=True).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    ce  = nn.CrossEntropyLoss()

    # Entrenamiento
    best_loss, bad = float('inf'), 0
    for ep in range(1, cfg.epochs + 1):
        net.train(); tl = 0
        for b, y in dl_tr:
            b = [t.to(dev) for t in b]
            y = y.to(dev)
            opt.zero_grad()
            loss = ce(net(b), y)
            loss.backward()
            opt.step()
            tl += loss.item() * y.size(0)
        tl /= len(tr)

        net.eval(); vl = cor = 0
        with torch.no_grad():
            for b, y in dl_va:
                b = [t.to(dev) for t in b]
                y = y.to(dev)
                out = net(b)
                vl += ce(out, y).item() * y.size(0)
                cor += (out.argmax(1) == y).sum().item()
        vl /= len(va)
        acc = 100 * cor / len(va)
        print(f"[ep {ep}/{cfg.epochs}] tl={tl:.3f} vl={vl:.3f} acc={acc:.1f}%")

        # Guardado del mejor modelo
        if vl < best_loss - 1e-4:
            best_loss, bad = vl, 0
            torch.save({'state_dict': net.state_dict(), 'num_classes': cfg.classes}, 'best_signnet.pt')
        else:
            bad += 1
            if bad >= cfg.patience:
                print('Early stopping')
                break

###############################################################################
# 7. CLI + CUDA por defecto
###############################################################################
if __name__ == '__main__':
    set_seed()
    ap = argparse.ArgumentParser(description='SignNet trainer (gForcePro+)')
    ap.add_argument('--root',    required=True)
    ap.add_argument('--classes', required=True, type=int)
    ap.add_argument('--batch',   type=int,   default=16)
    ap.add_argument('--epochs',  type=int,   default=50)
    ap.add_argument('--lr',      type=float, default=1e-3)
    ap.add_argument('--val',     type=float, default=0.2)
    ap.add_argument('--patience',type=int,   default=8)
    ap.add_argument('--workers', type=int,   default=4)
    ap.add_argument('--wd',      type=float, default=1e-4)
    ap.add_argument('--emg-rate',type=int,   default=500,
                   help='Frecuencia EMG para preprocesado (Hz)')
    ap.add_argument('--device', choices=['auto','cuda','cpu'], default='auto',
                   help="Dispositivo para entrenar: 'cuda','cpu' o 'auto' (por defecto cuda si está disponible)")
    args = ap.parse_args()

    # Selección de dispositivo
    if args.device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(args.device)
    print(f"Usando dispositivo: {dev}")

    cfg = CFG(
        root=args.root,
        classes=args.classes,
        batch=args.batch,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.lr,
        val=args.val,
        patience=args.patience,
        wd=args.wd,
        emg_rate=args.emg_rate
    )
    train(cfg, dev)
