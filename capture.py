#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import asyncio
import struct
import argparse
import msvcrt

import numpy as np
import h5py

from gforce import DataNotifFlags, GForceProfile, NotifDataType

# ----------------------------
# 1. Parámetros por defecto
# ----------------------------
DEFAULT_ADDRESS = "90:7B:C6:63:4C:B8"
DEFAULT_EMG_RATE = 500   # Hz
IMU_RATE         = 100   # Hz

# ------------------------------------
# 2. Callback y buffers globales
# ------------------------------------
buf_emg, buf_acc, buf_gyr, buf_euler, buf_quat = [], [], [], [], []

def on_data_raw(_, data: bytearray):
    ts = time.time()
    kind = data[0]
    p = data[1:]
    if kind == NotifDataType.NTF_EMG_ADC_DATA:
        for s in np.frombuffer(p, dtype=np.uint8).reshape(-1, 8):
            buf_emg.append((ts, *s))
    elif kind == NotifDataType.NTF_ACC_DATA:
        ax, ay, az = struct.unpack("<3h", p[:6])
        buf_acc.append((ts, ax, ay, az))
    elif kind == NotifDataType.NTF_GYO_DATA:
        gx, gy, gz = struct.unpack("<3h", p[:6])
        buf_gyr.append((ts, gx, gy, gz))
    elif kind == NotifDataType.NTF_EULER_DATA:
        roll, pitch, yaw = struct.unpack("<3f", p[:12])
        buf_euler.append((ts, roll, pitch, yaw))
    elif kind == NotifDataType.NTF_QUAT_FLOAT_DATA:
        qw, qx, qy, qz = struct.unpack("<4f", p[:16])
        buf_quat.append((ts, qw, qx, qy, qz))

def _sync_wait_space():
    while True:
        if msvcrt.getch() == b' ':
            break

async def wait_for_space(prompt: str):
    print(prompt)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _sync_wait_space)

# ------------------------------------------------
# 3. Almacenamiento en HDF5 con nueva estructura
# ------------------------------------------------
def process_and_store(gesture: str, session_id: int, sample: int):
    if not buf_emg or not buf_acc or not buf_gyr or not buf_euler or not buf_quat:
        print("❌ Datos insuficientes. Saltando guardado.")
        return

    emg_arr   = np.array(buf_emg)
    acc_arr   = np.array(buf_acc)
    gyr_arr   = np.array(buf_gyr)
    euler_arr = np.array(buf_euler)
    quat_arr  = np.array(buf_quat)

    # Directorio base: data/{gesture}
    out_dir = os.path.join("data", gesture)
    os.makedirs(out_dir, exist_ok=True)
    # Archivo: session_{SS}_sample_{NN}.h5
    out_path = os.path.join(
        out_dir,
        f"session_{session_id:02d}_sample_{sample:02d}.h5"
    )

    with h5py.File(out_path, 'w') as f:
        # Metadatos
        f.attrs['gesture']    = str(gesture)
        f.attrs['session_id'] = session_id
        f.attrs['sample']     = sample

        grp = f.create_group('raw')
        # Configurar chunks dinámicamente (filas, columnas)
        def create_ds(name, arr):
            rows, cols = arr.shape
            chunk_rows = min(1000, rows)
            grp.create_dataset(
                name,
                data=arr,
                compression='gzip',
                chunks=(chunk_rows, cols)
            )
        create_ds('emg',   emg_arr)
        create_ds('acc',   acc_arr)
        create_ds('gyro',  gyr_arr)
        create_ds('euler', euler_arr)
        create_ds('quat',  quat_arr)

    print(f"✔ Sample {sample} almacenada en {out_path}")

# ------------------------------------------------
# 4. Captura con calibración y control ESPACIO
# ------------------------------------------------
async def capture_loop(args):
    prof = GForceProfile()
    await prof.connect(args.address)

    print("Mantén el brazalete inmóvil para calibrar orientación...")
    await prof.calibrate_quaternion(duration=2.0)
    print("Calibración completada.")

    await prof.setEmgRawDataConfig(DEFAULT_EMG_RATE, 0xFF, 16, 8, cb=None, timeout=1000)
    flags = (
        DataNotifFlags.DNF_EMG_RAW
      | DataNotifFlags.DNF_ACCELERATE
      | DataNotifFlags.DNF_GYROSCOPE
      | DataNotifFlags.DNF_EULERANGLE
      | DataNotifFlags.DNF_QUATERNION
    )
    await prof.setDataNotifSwitch(flags, cb=None, timeout=1000)

    client = prof.device
    await client.start_notify(prof.notifyCharacteristic, on_data_raw)

    try:
        for sample in range(1, args.samples + 1):
            print(f"\n=== Muestra {sample}/{args.samples} ===")
            await wait_for_space("Presiona ESPACIO para iniciar...")
            buf_emg.clear(); buf_acc.clear()
            buf_gyr.clear(); buf_euler.clear(); buf_quat.clear()

            print("Grabando... presiona ESPACIO para detener.")
            await wait_for_space("")

            process_and_store(
                gesture    = args.gesture,
                session_id = args.session,
                sample     = sample
            )
    finally:
        await client.stop_notify(prof.notifyCharacteristic)
        await prof.stopDataNotification()
        try:
            await prof.disconnect()
        except asyncio.CancelledError:
            pass
        print("\nDesconectado limpiamente ✅")

# ------------------------------------
# 5. CLI y principal
# ------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Captura IMU + sEMG y almacena en HDF5 con nueva estructura de directorios"
    )
    p.add_argument("--address",  default=DEFAULT_ADDRESS,
                   help="BLE MAC/UUID del brazalete")
    p.add_argument("--gesture", required=True,
                   help="Nombre del gesto (obligatorio)")
    p.add_argument("--session",  type=int, required=True,
                   help="ID de sesión (obligatorio)")
    p.add_argument("--samples",  type=int, default=10,
                   help="Número de muestras (default: 10)")
    args = p.parse_args()

    asyncio.run(capture_loop(args))

if __name__ == "__main__":
    main()
