# trainer.py
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
from preprocessor import PreprocessLayer

# reproducibilidad
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

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
_pad=lambda x,T: F.pad(x,(0,0,0,T-x.size(0))) if x.size(0)<T else x

def collate(batch):
    xs, ys = zip(*batch)
    Te = max(s[0].size(0) for s in xs)
    Ti = max(max(s[j].size(0) for j in range(1,5)) for s in xs)
    out=[[] for _ in range(5)]
    for emg, acc, gyro, eul, quat in xs:
        out[0].append(_pad(emg,Te)); out[1].append(_pad(acc,Ti)); out[2].append(_pad(gyro,Ti)); out[3].append(_pad(eul,Ti)); out[4].append(_pad(quat,Ti))
    return tuple(torch.stack(z) for z in out), torch.tensor(ys)

###############################################################################
# 3. Preprocess
###############################################################################


###############################################################################
# 4. Model
###############################################################################
class Net(nn.Module):
    def __init__(self, nc, conv=(64,128)):
        super().__init__(); self.pre=PreprocessLayer(emg_rate=250,target_rate=100)
        self.conv = nn.Sequential(nn.Conv1d(21,conv[0],5,padding=2), nn.BatchNorm1d(conv[0]), nn.ReLU(),
                                  nn.Conv1d(conv[0],conv[1],5,padding=2), nn.BatchNorm1d(conv[1]), nn.ReLU())
        self.lstm=nn.LSTM(conv[1],64,2,batch_first=True,bidirectional=True,dropout=0.3)
        self.att = nn.Linear(128,1,bias=False); self.fc=nn.Linear(128,nc)
    def forward(self,b):
        x=self.pre(*b); x=self.conv(x).transpose(1,2); h,_=self.lstm(x)
        w=torch.softmax(self.att(h),1)
        return self.fc((w*h).sum(1))

###############################################################################
# 5. Config
###############################################################################
@dataclass
class CFG:
    root:str; classes:int; batch:int=16; workers:int=4; epochs:int=50; lr:float=1e-3; val:float=0.2; patience:int=8; wd:float=1e-4

###############################################################################
# 6. Train
###############################################################################

def _split(lst:List[str],r:float): random.shuffle(lst); k=int((1-r)*len(lst)); return lst[:k],lst[k:]

def train(cfg:CFG):
    patt=os.path.join(cfg.root,'session_*/user_*/phrase_*/rep_*/data.h5')
    files=sorted(glob.glob(patt))
    if not files: raise FileNotFoundError('No h5 found')
    tr,va=_split(files,cfg.val)
    pin=torch.cuda.is_available(); dev=torch.device('cuda' if pin else 'cpu')
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
    net=Net(cfg.classes).to(dev); opt=torch.optim.AdamW(net.parameters(),cfg.lr,weight_decay=cfg.wd); ce=nn.CrossEntropyLoss()
    best,bad=float('inf'),0
    for ep in range(1,cfg.epochs+1):
        net.train(); tl=0
        for b,y in dl_tr:
            b=[t.to(dev) for t in b]; y=y.to(dev)
            opt.zero_grad(); loss=ce(net(b),y); loss.backward(); opt.step(); tl+=loss.item()*y.size(0)
        tl/=len(tr)
        net.eval(); vl=cor=0
        with torch.no_grad():
            for b,y in dl_va:
                b=[t.to(dev) for t in b]; y=y.to(dev); log=net(b)
                vl+=ce(log,y).item()*y.size(0); cor+=(log.argmax(1)==y).sum().item()
        vl/=len(va); acc=100*cor/len(va)
        print(f'[ep {ep}/{cfg.epochs}] tl={tl:.3f} vl={vl:.3f} acc={acc:.1f}%')
        if vl<best-1e-4:
            best, bad = vl, 0
            torch.save({'state_dict':net.state_dict(),'num_classes':cfg.classes},'best_signnet.pt')
        else:
            bad+=1
            if bad>=cfg.patience: print('early stop'); break

###############################################################################
# 7. CLI
###############################################################################
if __name__=='__main__':
    ap=argparse.ArgumentParser(description='SignNet trainer (gForcePro+)')
    ap.add_argument('--root',   required=True)
    ap.add_argument('--classes',required=True,type=int)
    ap.add_argument('--batch',  type=int, default=16)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr',     type=float, default=1e-3)
    ap.add_argument('--val',    type=float, default=0.2)
    ap.add_argument('--patience',type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--wd',     type=float, default=1e-4)
    args=ap.parse_args()
    cfg=CFG(root=args.root, classes=args.classes, batch=args.batch, workers=args.workers,
            epochs=args.epochs, lr=args.lr, val=args.val, patience=args.patience, wd=args.wd)
    train(cfg)
