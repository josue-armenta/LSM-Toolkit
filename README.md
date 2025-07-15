# LSM‑Toolkit (v2025‑07)

**Pipeline de captura, entrenamiento y traducción de gestos de la *Lengua de Señas Mexicana* (LSM) usando el brazalete Oymotion gForce Pro+.**

<div align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/pytorch-%E2%89%A5%202.0-lightgrey" alt="PyTorch ≥ 2.0">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</div>

---

## Índice rápido
1. [Instalación](#instalación)
2. [Estructura de carpetas](#estructura-de-carpetas)
3. [Captura de datos](#captura-de-datos)
4. [Entrenamiento](#entrenamiento)
5. [Traducción en tiempo real](#traducción-en-tiempo-real)
6. [Modelo y preprocesamiento](#modelo-y-preprocesamiento)
7. [Aumento de datos](#aumento-de-datos)
8. [Preguntas frecuentes](#preguntas-frecuentes)
9. [Contribución](#contribución)
10. [Licencia](#licencia)

---

## Instalación

```bash
# 1. clona el repositorio
git clone https://github.com/josue-armenta/LSM-Toolkit.git
cd LSM-Toolkit

# 2. (opcional) crea un entorno virtual
python -m venv .venv && source .venv/bin/activate

# 3. instala dependencias
pip install -r requirements.txt   # incluye tsaug, pywavelets, bleak, etc.
```

> **Requisitos mínimos**  
> • Python ≥ 3.9 • PyTorch ≥ 2.0 (CUDA opcional) • gForce Pro+ vía BLE

---

## Estructura de carpetas

```
LSM-Toolkit/
├── capture.py          # grabación BLE → HDF5
├── trainer.py          # entrenamiento SignNet
├── translator.py       # inferencia continua
├── augment.py          # data‑augmentation tsaug
├── model.py            # CNN‑BiLSTM‑Attention (+Wavelets)
├── preprocessor.py     # alineación, resampleo, denoise
├── gforce.py           # wrapper BLE gForce SDK
├── requirements.txt
└── data/
    └── <gesture>/
        ├── session_01_sample_01.h5
        ├── session_01_sample_02.h5
        └── …
```

Cada archivo **`.h5`** contiene un grupo `raw/` con `emg`, `acc`, `gyro`, `euler`, `quat`
y los atributos HDF5:

| Atributo | Tipo | Descripción |
|----------|------|-------------|
| `gesture` | `str` | Nombre textual del gesto |
| `session_id` | `int` | ID de la sesión de captura |
| `sample` | `int` | N.º consecutivo dentro de la sesión |

---

## Captura de datos

```bash
python capture.py   --gesture hola   --session 1   --samples 15   --address 90:7B:C6:63:4C:B8
```

| Flag          | Tipo   | Predeterminado           | Descripción |
|---------------|--------|--------------------------|-------------|
| `--gesture`   | str    | — (obligatorio)          | Nombre del gesto |
| `--session`   | int    | — (obligatorio)          | ID de la sesión |
| `--samples`   | int    | `10`                     | Cantidad de capturas a grabar |
| `--address`   | str    | `90:7B:C6:63:4C:B8`      | MAC/UUID BLE gForce |

> *La frecuencia EMG se fija internamente a **500 Hz**; IMU a **100 Hz**.  
> Presiona **ESPACIO** para iniciar/detener cada muestra.*

---

## Entrenamiento

```bash
python trainer.py   --root data   --gestures hola adios gracias   --epochs 60   --batch 32   --device auto
```

Argumentos principales:

| Flag          | Tipo        | Predeterminado | Descripción |
|---------------|-------------|---------------|-------------|
| `--root`      | str         | `data`        | Carpeta raíz del dataset |
| `--gestures`  | lista str   | *infiere*     | Lista de gestos (orden) |
| `--epochs`    | int         | `50`          | Épocas de entrenamiento |
| `--batch`     | int         | `16`          | Tamaño de lote |
| `--lr`        | float       | `1e-3`        | Learning‑rate |
| `--val`       | float       | `0.2`         | Proporción validación |
| `--patience`  | int         | `8`           | Early‑stopping |
| `--device`    | `cuda/cpu/auto` | `auto`   | Dispositivo de cómputo |

Genera **`best_signnet.pt`** que incluye pesos y lista de gestos –
no necesitas un archivo `labels.txt` separado.

---

## Traducción en tiempo real

```bash
python translator.py   --model best_signnet.pt   --address 90:7B:C6:63:4C:B8   -w 7   -t 0.8   -s ma   --interval 0.15   --debug
```

| Flag / alias      | Descripción |
|-------------------|-------------|
| `--model`         | Checkpoint entrenado |
| `--interval`      | Segundos entre inferencias |
| `-w/--window-size`| Tamaño de ventana para post‑procesado |
| `-t/--conf-thresh`| Umbral de confianza (0–1) |
| `-s/--smoother`   | `ma` media‑móvil ó `vote` |
| `--debug`         | Imprime info detallada |

---

## Modelo y preprocesamiento

```text
┌───────────────┐   Wavelet denoise (opcional)
│ PreprocessLayer│──► Resample EMG →100 Hz
└───────────────┘       Concatenación EMG+IMU = 21ch
        │
   1‑D CNN ×2
        │
     Bi‑LSTM ×2
        │
  Atención global
        │
   Softmax cls.
```

Parámetros clave (`model.py`):

| Parámetro     | Valor por defecto |
|---------------|------------------|
| `conv_ch`     | `(64, 128)`      |
| `lstm_hidden` | `64`             |
| `use_wavelets`| `False`          |

---

## Aumento de datos

```python
from augment import AUG_PIPE
```

*Pipeline `tsaug` aplicado **solo** en entrenamiento*:

- `AddNoise(scale=0.02)`  
- `TimeWarp(max_speed_ratio=1.1)`  
- `Drift(max_drift=(0.05, 0.05))`

Desactívalo pasando `augment=False` en `GForceTrialDataset`.

---

## Preguntas frecuentes

| Problema | Solución |
|----------|----------|
| **No conecta el brazalete** | Verifica permisos BLE y la MAC (`bluetoothctl devices`). |
| **`pywavelets` no se instala** | `pip install --only-binary :all: pywavelets` o instala *build‑essential*. |
| **CUDA no detectada** | Instala el *wheel* de PyTorch con soporte `+cu118` y ejecuta `trainer.py --device cuda`. |

---

## Contribución

1. Haz **fork** y crea una rama descriptiva.  
2. Sigue PEP‑8 y añade pruebas si procede.  
3. Abre un **pull request** detallando cambios y motivación.

---

## Licencia

Distribuido bajo licencia **MIT** – consulta `LICENSE` para más información.
