#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translator.py – Streaming continuo y depuración
Con suavizado/votación sobre N predicciones y umbral de confianza
Autor: ChatGPT – julio 2025
"""

import argparse
import asyncio
import collections
import struct
import time
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from model import SignNet
from gforce import DataNotifFlags, GForceProfile, NotifDataType

# Parámetros de señal y buffers
DEFAULT_ADDRESS = "90:7B:C6:63:4C:B8"
DEFAULT_MODEL   = "best_signnet.pt"
EMG_RATE   = 500   # Hz
IMU_RATE   = 100   # Hz
WINDOW_SEC = 0.8
EMG_LEN    = int(WINDOW_SEC * EMG_RATE)
IMU_LEN    = int(WINDOW_SEC * IMU_RATE)

buf_emg:   Deque[tuple] = collections.deque(maxlen=EMG_LEN)
buf_acc:   Deque[tuple] = collections.deque(maxlen=IMU_LEN)
buf_gyr:   Deque[tuple] = collections.deque(maxlen=IMU_LEN)
buf_euler: Deque[tuple] = collections.deque(maxlen=IMU_LEN)
buf_quat:  Deque[tuple] = collections.deque(maxlen=IMU_LEN)

def on_data_raw(_, data: bytearray):
    """Callback BLE → buffers."""
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

def load_model(path: str, device) -> Tuple[SignNet, int, Optional[List[str]]]:
    """
    Carga un checkpoint o módulo completo y extrae:
      - model: instancia de SignNet en modo eval
      - n_cls: número de clases
      - labels (opcional): lista de nombres de gestos
    """
    obj = torch.load(path, map_location=device)
    labels: Optional[List[str]] = None

    # Caso 1: checkpoint serializado como módulo completo
    if isinstance(obj, nn.Module):
        model = obj.to(device)
        model.eval()
        n_cls = model.cls.out_features if hasattr(model, 'cls') else None
        if hasattr(model, 'gestures'):
            labels = getattr(model, 'gestures')
        return model, n_cls, labels

    # Caso 2: checkpoint serializado como dict
    if isinstance(obj, dict):
        if 'gestures' in obj:
            labels = obj['gestures']

        # Detecta dónde está el state_dict y num_classes
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

        # Renombra capas antiguas si hace falta
        rename_map = {
            'att.weight': 'attn.weight',
            'fc.weight':  'cls.weight',
            'fc.bias':    'cls.bias',
        }
        for old, new in rename_map.items():
            if old in state_dict and new not in state_dict:
                state_dict[new] = state_dict.pop(old)

        # Reconstruye el modelo y carga pesos
        model = SignNet(n_cls).to(device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠ Pesos faltantes: {missing}")
        if unexpected:
            print(f"⚠ Pesos inesperados: {unexpected}")
        model.eval()
        return model, n_cls, labels

    # Si no es ni Module ni dict
    raise RuntimeError('Checkpoint no es ni Module ni dict')

async def inference_loop(
    model, device, labels,
    interval: float,
    debug: bool,
    window_size: int,
    conf_thresh: float,
    smoother: str,
    alpha: float
):
    """
    Ejecuta inferencias cada `interval` s, acumula las últimas
    `window_size` predicciones y aplica:
      - media móvil (“ma”)
      - o voto mayoritario (“vote”)
    Sólo emite si la confianza ≥ conf_thresh.
    """
    prob_buffer = collections.deque(maxlen=window_size)
    ema_probs = None
    pred_buffer = collections.deque(maxlen=window_size)
    conf_buffer = collections.deque(maxlen=window_size)
    prev_out = None

    print(f"➡️ Post-proceso: método={smoother}, ventana={window_size}, umbral={conf_thresh}")

    while True:
        await asyncio.sleep(interval)

        # Verifica que haya datos suficientes
        if min(len(buf_emg), len(buf_acc), len(buf_gyr),
               len(buf_euler), len(buf_quat)) < min(EMG_LEN, IMU_LEN):
            continue

        # Construye el batch
        emg_np   = np.array(buf_emg,   dtype=np.float32)[-EMG_LEN:, 1:]
        if emg_np.var() < 1.0:
            if debug:
                print("\r⚠ Var(emg)<1 – señal vacía", end="", flush=True)
            continue

        acc_np   = np.array(buf_acc,   dtype=np.float32)[-IMU_LEN:, 1:]
        gyr_np   = np.array(buf_gyr,   dtype=np.float32)[-IMU_LEN:, 1:]
        euler_np = np.array(buf_euler, dtype=np.float32)[-IMU_LEN:, 1:]
        quat_np  = np.array(buf_quat,  dtype=np.float32)[-IMU_LEN:, 1:]

        batch = [
            torch.from_numpy(x).unsqueeze(0).to(device)
            for x in (emg_np, acc_np, gyr_np, euler_np, quat_np)
        ]
        with torch.no_grad():
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1)[0]
            pred   = int(probs.argmax().item())
            conf   = float(probs[pred].item())

        prob_buffer.append(probs)
        pred_buffer.append(pred)
        conf_buffer.append(conf)
        
        # ---------- NEW: update EMA ----------
        if smoother == "ema":
            if ema_probs is None:
                ema_probs = probs.clone()
            else:
                ema_probs = alpha * probs + (1 - alpha) * ema_probs
        
        if len(pred_buffer) < window_size:
            continue

        # Selección post-proceso
        if smoother == "ema":
            out_probs = ema_probs
            out_pred  = int(out_probs.argmax().item())
            out_conf  = float(out_probs[out_pred].item())
        elif smoother == "ma":
            avg_probs = torch.stack(list(prob_buffer), dim=0).mean(dim=0)
            out_pred = int(avg_probs.argmax().item())
            out_conf = float(avg_probs[out_pred].item())
        else:  # voto mayoritario
            counts = collections.Counter(pred_buffer)
            maj_pred, count = counts.most_common(1)[0]
            if count > window_size // 2:
                idxs = [i for i, p in enumerate(pred_buffer) if p == maj_pred]
                out_pred = maj_pred
                out_conf = sum(conf_buffer[i] for i in idxs) / len(idxs)
            else:
                continue  # sin mayoría clara

        # Aplica umbral de confianza
        if out_conf < conf_thresh:
            continue

        label = labels[out_pred] if labels else str(out_pred)

        # Imprime sólo cuando cambia la clase (o en modo debug)
        if debug:
            if smoother == "ma":
                preds_list = [int(p.argmax().item()) for p in prob_buffer]
                print(f"\r[MA] preds={preds_list} → {out_pred} ({out_conf:.2f})", end="", flush=True)
            else:
                print(f"\r[VOTE] preds={list(pred_buffer)} counts={counts} → {out_pred} ({out_conf:.2f})", end="", flush=True)
        elif out_pred != prev_out:
            print(f"\r⏩ {label:<15} ({out_conf:.2f})            ", end="", flush=True)
            prev_out = out_pred

async def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, n_cls, labels = load_model(args.model, device)

    prof = GForceProfile()
    print("🔌 Conectando a", args.address, "…")
    await prof.connect(args.address)
    print("📐 Calibrando orientación (2 s)…")
    await prof.calibrate_quaternion(duration=2.0)
    await prof.setEmgRawDataConfig(EMG_RATE, 0xFF, 16, 8, cb=None, timeout=1000)
    flags = (
        DataNotifFlags.DNF_EMG_RAW |
        DataNotifFlags.DNF_ACCELERATE |
        DataNotifFlags.DNF_GYROSCOPE |
        DataNotifFlags.DNF_EULERANGLE |
        DataNotifFlags.DNF_QUATERNION
    )
    await prof.setDataNotifSwitch(flags, cb=None, timeout=1000)
    await prof.device.start_notify(prof.notifyCharacteristic, on_data_raw)
    print("🟢 Streaming – haz un gesto (Ctrl-C para salir)")

    try:
        await inference_loop(
            model, device, labels,
            args.interval, args.debug,
            args.window_size, args.conf_thresh, args.smoother, args.alpha
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n🛑 Terminando…")
    finally:
        await prof.device.stop_notify(prof.notifyCharacteristic)
        await prof.stopDataNotification()
        await prof.disconnect()

def main():
    p = argparse.ArgumentParser(
        description="Inferencia LSM gForce Pro+ – stream contínuo con suavizado/votación"
    )
    p.add_argument("--address", default=DEFAULT_ADDRESS,
                   help="BLE MAC/UUID del brazalete")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Ruta al checkpoint .pt")
    p.add_argument("--interval", type=float, default=0.2,
                   help="Segundos entre inferencias")
    p.add_argument("--debug", action="store_true",
                   help="Muestra info de depuración")
    p.add_argument("-w", "--window-size", type=int, default=5,
                   help="Tamaño de ventana para MA/votación")
    p.add_argument("-t", "--conf-thresh", type=float, default=0.75,
                   help="Umbral de confianza mínimo para emitir clase")
    p.add_argument("--smoother", choices=["ma","vote","ema"], default="ema",
                   help="Método de post‑proceso")
    p.add_argument("--alpha", type=float, default=0.2,
                   help="Factor de olvido de la EMA (0<α≤1)")
    args = p.parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
