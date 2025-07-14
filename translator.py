#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translator.py – Streaming continuo y depuración  
Autor: ChatGPT – julio 2025

Conecta al brazalete **gForce Pro+**, inicia automáticamente el streaming tras la
calibración y muestra en tiempo real el gesto más probable usando el modelo
`best_signnet.pt`. Ahora usa tasa EMG fija en 500 Hz y extrae las etiquetas
directamente del checkpoint (clave `'gestures'`).
"""

import argparse
import asyncio
import collections
import struct
import time
from typing import Deque, List, Optional

import numpy as np
import torch
import torch.nn as nn

from model import SignNet
from gforce import DataNotifFlags, GForceProfile, NotifDataType

# Parámetros de señal y buffers
DEFAULT_ADDRESS = "90:7B:C6:63:4C:B8"
DEFAULT_MODEL   = "best_signnet.pt"
EMG_RATE   = 500   # Hz (constante fija)
IMU_RATE   = 100   # Hz
WINDOW_SEC = 0.8
EMG_LEN    = int(WINDOW_SEC * EMG_RATE)
IMU_LEN    = int(WINDOW_SEC * IMU_RATE)

buf_emg:   Deque[tuple] = collections.deque(maxlen=EMG_LEN)
buf_acc:   Deque[tuple] = collections.deque(maxlen=IMU_LEN)
buf_gyr:   Deque[tuple] = collections.deque(maxlen=IMU_LEN)
buf_euler: Deque[tuple] = collections.deque(maxlen=IMU_LEN)
buf_quat:  Deque[tuple] = collections.deque(maxlen=IMU_LEN)

# Callback BLE → buffers
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

# Carga de modelo y extracción de etiquetas desde el checkpoint
def load_model(path: str, device):
    obj = torch.load(path, map_location=device)
    labels: Optional[List[str]] = None

    # Caso 1: modelo serializado completo
    if isinstance(obj, nn.Module):
        model = obj.to(device)
        model.eval()
        n_cls = model.cls.out_features if hasattr(model, 'cls') else None
        if hasattr(model, 'gestures'):
            labels = getattr(model, 'gestures')
        return model, n_cls, labels

    # Caso 2: diccionario de checkpoint
    if isinstance(obj, dict):
        # Extrae etiquetas si vienen en 'gestures'
        if 'gestures' in obj:
            labels = obj['gestures']

        # Determina state_dict y num_classes
        if 'cls.weight' in obj:
            state_dict = obj
            n_cls = state_dict['cls.weight'].shape[0]
        elif 'state_dict' in obj:
            state_dict = obj['state_dict']
            n_cls = obj.get('num_classes') or state_dict['cls.weight'].shape[0]
        elif 'model_state_dict' in obj:
            state_dict = obj['model_state_dict']
            n_cls = state_dict['cls.weight'].shape[0]
        else:
            raise RuntimeError('Formato de checkpoint no reconocido')
    else:
        raise RuntimeError('Checkpoint no es ni Module ni dict')

    # Adaptar nombres de capas antiguas si hace falta
    rename_map = {
        'att.weight':    'attn.weight',
        'fc.weight':     'cls.weight',
        'fc.bias':       'cls.bias',
    }
    for old, new in rename_map.items():
        if old in state_dict and new not in state_dict:
            state_dict[new] = state_dict.pop(old)

    # Reconstruye y carga pesos
    model = SignNet(n_cls).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠ Pesos faltantes: {missing}")
    if unexpected:
        print(f"⚠ Pesos inesperados: {unexpected}")
    model.eval()
    return model, n_cls, labels

# Bucle de inferencia (sin cambios)
async def inference_loop(model, device, labels, interval, debug):
    prev_pred = None
    while True:
        await asyncio.sleep(interval)
        if min(len(buf_emg), len(buf_acc), len(buf_gyr),
               len(buf_euler), len(buf_quat)) < min(EMG_LEN, IMU_LEN):
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

        batch = [torch.from_numpy(x).unsqueeze(0).to(device)
                 for x in (emg_np, acc_np, gyr_np, euler_np, quat_np)]
        with torch.no_grad():
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1)[0]
            pred   = int(probs.argmax().item())
            conf   = float(probs[pred].item())

        label = labels[pred] if labels else str(pred)

        if debug:
            var_emg = float(emg_np.var())
            top_vals, top_idx = torch.topk(probs, k=min(3, probs.size(0)))
            top_str = " ".join(
                f"{labels[i] if labels else i}:{p:.2f}"
                for p, i in zip(top_vals, top_idx)
            )
            print(f"\rvar:{var_emg:5.1f}  {top_str:<30}",
                  end="", flush=True)
        elif pred != prev_pred:
            print(f"\r⏩ {label:<15} ({conf:.2f})            ",
                  end="", flush=True)
            prev_pred = pred

# Rutina principal
async def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, n_cls, labels = load_model(args.model, device)

    prof = GForceProfile()
    print("🔌 Conectando a", args.address, "…")
    await prof.connect(args.address)

    print("📐 Calibrando orientación (2 s)…")
    await prof.calibrate_quaternion(duration=2.0)

    await prof.setEmgRawDataConfig(EMG_RATE, 0xFF, 16, 8,
                                   cb=None, timeout=1000)
    flags = (DataNotifFlags.DNF_EMG_RAW |
             DataNotifFlags.DNF_ACCELERATE |
             DataNotifFlags.DNF_GYROSCOPE |
             DataNotifFlags.DNF_EULERANGLE |
             DataNotifFlags.DNF_QUATERNION)
    await prof.setDataNotifSwitch(flags, cb=None, timeout=1000)
    await prof.device.start_notify(prof.notifyCharacteristic,
                                   on_data_raw)

    print("🟢 Streaming – haz un gesto (Ctrl-C para salir)")

    try:
        await inference_loop(model, device, labels,
                             args.interval, args.debug)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n🛑 Terminando…")
    finally:
        await prof.device.stop_notify(prof.notifyCharacteristic)
        await prof.stopDataNotification()
        await prof.disconnect()

# CLI
def main():
    p = argparse.ArgumentParser(
        description="Inferencia LSM gForce Pro+ – stream continuo"
    )
    p.add_argument("--address", default=DEFAULT_ADDRESS,
                   help="BLE MAC/UUID del brazalete")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Ruta al checkpoint .pt")
    p.add_argument("--interval", type=float, default=0.2,
                   help="Segundos entre inferencias")
    p.add_argument("--debug", action="store_true",
                   help="Muestra info de depuración")
    args = p.parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
