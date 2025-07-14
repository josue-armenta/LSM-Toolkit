#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""capture.py – Streaming continuo y depuración
Autor: ChatGPT – julio 2025

Conecta al brazalete **gForce Pro+**, inicia automáticamente el *streaming* tras la
calibración y muestra en tiempo real el gesto más probable usando el modelo
`best_signnet.pt`.  Corrige los problemas anteriores:

* 🛠️ **Labels > clases**: si `labels.txt` tiene más nombres que salidas del
  modelo (p.ej. 50 vs 3), se recorta a `n_clases` y se avisa en consola.
* 🛠️ **Código limpio**: elimina duplicados y errores de sintaxis en el bucle de
  inferencia.
* 🛠️ **Top‑k debug**: con `--debug` se muestran las 3 clases más probables.
"""

###############################################################################
# Imports
###############################################################################

import argparse
import asyncio
import collections
import os
import struct
import time
from typing import Deque, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessor import PreprocessLayer
from gforce import DataNotifFlags, GForceProfile, NotifDataType

###############################################################################
# Parámetros de señal y buffers
###############################################################################

DEFAULT_ADDRESS = "90:7B:C6:63:4C:B8"
DEFAULT_MODEL   = "best_signnet.pt"
EMG_RATE   = 500  # Hz
IMU_RATE   = 100  # Hz
WINDOW_SEC = 0.8
EMG_LEN    = int(WINDOW_SEC * EMG_RATE)
IMU_LEN    = int(WINDOW_SEC * IMU_RATE)

buf_emg: Deque[tuple]   = collections.deque(maxlen=EMG_LEN)
buf_acc: Deque[tuple]   = collections.deque(maxlen=IMU_LEN)
buf_gyr: Deque[tuple]   = collections.deque(maxlen=IMU_LEN)
buf_euler: Deque[tuple] = collections.deque(maxlen=IMU_LEN)
buf_quat: Deque[tuple]  = collections.deque(maxlen=IMU_LEN)

###############################################################################
# Callback BLE → buffers
###############################################################################

def on_data_raw(_, data: bytearray):
    ts = time.time()
    k  = data[0]
    p  = data[1:]
    if k == NotifDataType.NTF_EMG_ADC_DATA:
        for s in np.frombuffer(p, dtype=np.uint8).reshape(-1, 8):
            buf_emg.append((ts, *s))
    elif k == NotifDataType.NTF_ACC_DATA:
        buf_acc.append((ts, *struct.unpack("<3h", p[:6])))
    elif k == NotifDataType.NTF_GYO_DATA:
        buf_gyr.append((ts, *struct.unpack("<3h", p[:6])))
    elif k == NotifDataType.NTF_EULER_DATA:
        buf_euler.append((ts, *struct.unpack("<3f", p[:12])))
    elif k == NotifDataType.NTF_QUAT_FLOAT_DATA:
        buf_quat.append((ts, *struct.unpack("<4f", p[:16])))

###############################################################################
# Modelo (extraído de trainer.py)
###############################################################################


class SignNet(nn.Module):
    def __init__(self, num_classes, conv_ch=(64, 128)):
        super().__init__()
        self.prep = PreprocessLayer(emg_rate=EMG_RATE,target_rate=100)
        self.conv = nn.Sequential(
            nn.Conv1d(21, conv_ch[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_ch[0]), nn.ReLU(),
            nn.Conv1d(conv_ch[0], conv_ch[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_ch[1]), nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_ch[1], 64, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.attn = nn.Linear(128, 1, bias=False)
        self.cls  = nn.Linear(128, num_classes)

    def forward(self, batch):
        emg, acc, gyro, euler, quat = batch
        x = self.prep(emg, acc, gyro, euler, quat)
        x = self.conv(x).transpose(1, 2)
        h, _ = self.lstm(x)
        alpha = torch.softmax(self.attn(h), dim=1)
        ctx = (alpha * h).sum(dim=1)
        return self.cls(ctx)

###############################################################################
# Carga de modelo y etiquetas
###############################################################################

def load_model(path, device):
    """Carga distintos formatos de checkpoint.

    * **state_dict plano:**   contiene directamente las capas.
    * **dict con 'state_dict':**  tu nuevo formato `{state_dict: …, num_classes: k}`.
    * **dict con 'model_state_dict':** estilo Torch Lightning.
    * **modelo completo:**   fue serializado con `torch.save(model, …)`.
    """
    obj = torch.load(path, map_location=device)

    # Caso 1: guardado completo → devuelve tal cual
    if isinstance(obj, nn.Module):
        model = obj.to(device)
        model.eval()
        n_cls = model.cls.out_features if hasattr(model, 'cls') else None
        return model, n_cls

    # Caso 2: diccionario conteniendo state_dict
    if isinstance(obj, dict):
        if 'cls.weight' in obj:              # state_dict plano
            state_dict = obj
            n_cls = state_dict['cls.weight'].shape[0]
        elif 'state_dict' in obj:
            state_dict = obj['state_dict']   # tu nuevo formato
            n_cls = obj.get('num_classes')
            if n_cls is None:
                n_cls = state_dict['cls.weight'].shape[0]
        elif 'model_state_dict' in obj:
            state_dict = obj['model_state_dict']  # Lightning
            n_cls = state_dict['cls.weight'].shape[0]
        else:
            raise RuntimeError('Formato de checkpoint no reconocido')
    else:
        raise RuntimeError('Checkpoint no es ni Module ni dict')

        # Adaptar nombres si vienen de una versión anterior
    rename_map = {
        'att.weight': 'attn.weight',  # antiguo -> nuevo
        'fc.weight':  'cls.weight',
        'fc.bias':    'cls.bias',
    }
    for old, new in rename_map.items():
        if old in state_dict and new not in state_dict:
            state_dict[new] = state_dict.pop(old)

    # Reconstruye modelo
    model = SignNet(n_cls).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠ Pesos faltantes: {missing}")
    if unexpected:
        print(f"⚠ Pesos inesperados: {unexpected}")
    model.eval()
    return model, n_cls

def load_labels(path, n_cls):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        labels = [l.strip() for l in fh if l.strip()]
    if len(labels) > n_cls:
        print(f"⚠ labels.txt tiene {len(labels)} nombres pero el modelo solo {n_cls} clases; se truncará.")
        labels = labels[:n_cls]
    elif len(labels) < n_cls:
        print(f"⚠ labels.txt tiene menos nombres ({len(labels)}) que clases ({n_cls}); se ignorarán nombres.")
        return None
    return labels

###############################################################################
# Bucle de inferencia
###############################################################################

async def inference_loop(model, device, labels, interval, debug):
    prev_pred = None
    n_labels = len(labels) if labels else None
    while True:
        await asyncio.sleep(interval)
        if min(len(buf_emg), len(buf_acc), len(buf_gyr), len(buf_euler), len(buf_quat)) < min(EMG_LEN, IMU_LEN):
            continue

        emg_np   = np.array(buf_emg,   dtype=np.float32)[-EMG_LEN:, 1:]
        if emg_np.var() < 1.0:
            if debug:
                print("\r⚠ Var(emg)<1 – señal vacía", end="", flush=True)
            continue

        acc_np   = np.array(buf_acc,   dtype=np.float32)[-IMU_LEN:, 1:]
        gyr_np   = np.array(buf_gyr,   dtype=np.float32)[-IMU_LEN:, 1:]
        euler_np = np.array(buf_euler, dtype=np.float32)[-IMU_LEN:, 1:]
        quat_np  = np.array(buf_quat,  dtype=np.float32)[-IMU_LEN:, 1:]

        batch = [torch.from_numpy(x).unsqueeze(0).to(device) for x in (emg_np, acc_np, gyr_np, euler_np, quat_np)]
        with torch.no_grad():
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1)[0]
            pred   = int(probs.argmax().item())
            conf   = float(probs[pred].item())

        label = labels[pred] if labels else str(pred)

        if debug:
            var_emg = float(emg_np.var())
            top_vals, top_idx = torch.topk(probs, k=min(3, probs.size(0)))
            top_str = " ".join(f"{(labels[i] if labels else i)}:{p:.2f}" for p, i in zip(top_vals, top_idx))
            print(f"\rvar:{var_emg:5.1f}  {top_str:<30}", end="", flush=True)
        elif pred != prev_pred:
            print(f"\r⏩ {label:<15} ({conf:.2f})            ", end="", flush=True)
            prev_pred = pred

###############################################################################
# Rutina principal
###############################################################################

async def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, n_cls = load_model(args.model, device)
    labels = load_labels(args.labels, n_cls)

    prof = GForceProfile()
    print("🔌 Conectando a", args.address, "…")
    await prof.connect(args.address)

    print("📐 Calibrando orientación (2 s)…")
    await prof.calibrate_quaternion(duration=2.0)

    await prof.setEmgRawDataConfig(EMG_RATE, 0xFF, 16, 8, cb=None, timeout=1000)
    flags = (DataNotifFlags.DNF_EMG_RAW | DataNotifFlags.DNF_ACCELERATE |
             DataNotifFlags.DNF_GYROSCOPE | DataNotifFlags.DNF_EULERANGLE |
             DataNotifFlags.DNF_QUATERNION)
    await prof.setDataNotifSwitch(flags, cb=None, timeout=1000)
    await prof.device.start_notify(prof.notifyCharacteristic, on_data_raw)

    print("🟢 Streaming – haz un gesto (Ctrl‑C para salir)")

    try:
        await inference_loop(model, device, labels, args.interval, args.debug)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n🛑 Terminando…")
    finally:
        await prof.device.stop_notify(prof.notifyCharacteristic)
        await prof.stopDataNotification()
        await prof.disconnect()

###############################################################################
# CLI
###############################################################################

def main():
    p = argparse.ArgumentParser(description="Inferencia LSM gForce Pro+ – stream continuo")
    p.add_argument("--address", default=DEFAULT_ADDRESS, help="BLE MAC/UUID del brazalete")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ruta al checkpoint .pt")
    p.add_argument("--labels", default="labels.txt", help="Archivo con nombres de clase")
    p.add_argument("--interval", type=float, default=0.2, help="Segundos entre inferencias")
    p.add_argument("--debug", action="store_true", help="Muestra info de depuración")
    p.add_argument("--emg-rate", type=int, default=EMG_RATE, help="Frecuencia EMG (Hz) para preprocesado")
    args = p.parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
