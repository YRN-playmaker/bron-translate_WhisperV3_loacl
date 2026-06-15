# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Quick GPU sanity check for BroN-translate (Local Edition).
# Imports main first so the Windows NVIDIA DLL paths get registered exactly the
# way they are at real startup, then loads the tiny Whisper model on the GPU.
#
#   python test_gpu.py
#
# Prints "GPU OK" if the CUDA / cuBLAS / cuDNN chain works. If this passes but
# large-v3 fails, the problem is VRAM or model path, not your GPU.

import main  # noqa: F401  (triggers _register_nvidia_dll_dirs on Windows)
import numpy as np
import faster_whisper

print("[*] Loading tiny model on GPU (float16)...")
m = faster_whisper.WhisperModel("tiny", device="cuda", compute_type="float16")
list(m.transcribe(np.zeros(16000, dtype=np.float32))[0])
print("GPU OK")
