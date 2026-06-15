# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# BroN-translate (Local Edition)
# Copyright (C) 2026 YRN-playmaker
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version. This program is distributed WITHOUT ANY WARRANTY; see the
# GNU AGPL for details: <https://www.gnu.org/licenses/agpl-3.0.html>.
#
# DUAL LICENSING: the AGPL requires that any modified or networked deployment
# also be released under the AGPL. If you want to use this software in a closed-
# source or commercial product without that obligation, a separate commercial
# license is available — contact 2213993473a@gmail.com.
"""
BroN-translate (Local Edition)
==============================
A self-hosted, single-process real-time speech translation server.

Pipeline:  microphone PCM  ->  [optional Vosk live preview]
                            ->  VAD segmentation
                            ->  faster-whisper (large-v3) final transcription
                            ->  DeepSeek translation
                            ->  captions over WebSocket

This edition has NO accounts, NO billing, NO gateway. It runs entirely on
127.0.0.1 against your own GPU. Configure it through a local .env file.

NOTE: two extension hooks (`refine_segment_audio` and
`postprocess_transcript`) are intentionally left as pass-throughs. They are
reserved seams for a separate research build and do nothing here.
"""

import os
import io
import sys
import json
import wave
import asyncio
import webbrowser
from contextlib import asynccontextmanager


def _register_nvidia_dll_dirs():
    """On Windows, faster-whisper/CTranslate2 needs cuBLAS + cuDNN DLLs but does
    not look inside the pip-installed `nvidia-*-cu12` packages on its own. We add
    those bin folders to the DLL search path so `import faster_whisper` + GPU
    inference can find cublas64_12.dll / cudnn*.dll.

    Install the runtime once with:
        pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
    No-op on non-Windows or if the packages aren't present.
    """
    if sys.platform != "win32":
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if not spec or not spec.submodule_search_locations:
            return
        nvidia_root = list(spec.submodule_search_locations)[0]
    except Exception:
        return
    for sub in ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"):
        bin_dir = os.path.join(nvidia_root, sub, "bin")
        if os.path.isdir(bin_dir):
            try:
                os.add_dll_directory(bin_dir)
            except Exception:
                pass
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


_register_nvidia_dll_dirs()

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from openai import AsyncOpenAI

load_dotenv()

# ==========================================================================
# Configuration (everything is read from .env — see .env.example)
# ==========================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

# Whisper model. Default large-v3 (full 32-layer decoder).
#   - "large-v3"        : best accuracy, trustworthy token-level confidence.
#   - "large-v3-turbo"  : ~4x faster, distilled 4-layer decoder. Faster but
#                         its confidence signals are less reliable — use only
#                         as a speed/ablation comparison, not as the default.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3").strip()
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda").strip()          # cuda | cpu
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16").strip()  # float16 | int8_float16 | int8

# Vosk model directory (only loaded if a client enables live preview).
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "model/vosk-model-small-en-us").strip()

HOST = os.getenv("HOST", "127.0.0.1").strip()
PORT = int(os.getenv("PORT", "8000"))
AUTO_OPEN_BROWSER = os.getenv("AUTO_OPEN_BROWSER", "1").strip() not in ("0", "false", "False", "")

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # int16

# ==========================================================================
# Lazy-loaded engines
# ==========================================================================
_whisper_model = None
_vosk_model = None
deepseek_client = None

LANG_NAMES = {"zh": "中文", "en": "英文", "ja": "日文", "ru": "俄文", "ms": "马来语", "auto": "auto"}


def get_whisper():
    """Load faster-whisper on first use. Heavy: pulls the model into VRAM."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print(f"[*] Loading faster-whisper '{WHISPER_MODEL}' on {WHISPER_DEVICE} "
              f"({WHISPER_COMPUTE_TYPE}) ... first run downloads the weights.")
        _whisper_model = WhisperModel(
            WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
        print("[*] Whisper ready.")
    return _whisper_model


def get_vosk():
    """Load Vosk on first use. Only needed for the live-preview captions."""
    global _vosk_model
    if _vosk_model is None:
        from vosk import Model as VoskModel
        if not os.path.isdir(VOSK_MODEL_PATH):
            raise FileNotFoundError(
                f"Vosk model not found at '{VOSK_MODEL_PATH}'. "
                "Live preview needs a Vosk model — download one from "
                "https://alphacephei.com/vosk/models and unzip it there, "
                "or just leave live preview off."
            )
        print(f"[*] Loading Vosk model from '{VOSK_MODEL_PATH}' ...")
        _vosk_model = VoskModel(VOSK_MODEL_PATH)
        print("[*] Vosk ready.")
    return _vosk_model


# ==========================================================================
# Research extension hooks — INTENTIONALLY INERT in this edition.
# They exist so a separate build can slot logic in without changing the
# pipeline around them. Here they are pure pass-throughs.
# ==========================================================================
def refine_segment_audio(pcm_int16: np.ndarray) -> np.ndarray:
    """Hook: pre-process a speech segment's audio before transcription.
    Reserved for a research extension; returns the audio unchanged."""
    return pcm_int16


def postprocess_transcript(text: str, segments_meta: list) -> str:
    """Hook: post-process a transcription result before translation.
    `segments_meta` carries per-segment fields exposed by the recognizer.
    Reserved for a research extension; returns the text unchanged."""
    return text


# ==========================================================================
# ASR: transcribe one finished speech segment with faster-whisper
# ==========================================================================
def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    audio_int16 = refine_segment_audio(audio_int16)  # inert hook
    return audio_int16.astype(np.float32) / 32768.0


def transcribe_segment(pcm_bytes: bytes, source_lang: str) -> str:
    """Run Whisper on a single segment. Blocking — call via asyncio.to_thread."""
    if not pcm_bytes:
        return ""
    audio = _pcm_to_float32(pcm_bytes)
    model = get_whisper()
    language = None if source_lang in ("auto", "", None) else source_lang
    segments, _info = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=False,  # we do our own VAD upstream
    )
    parts, meta = [], []
    for seg in segments:
        parts.append(seg.text)
        # Per-segment fields Whisper exposes (kept here so the postprocess
        # hook has something to work with later; unused in this edition).
        meta.append({
            "text": seg.text,
            "avg_logprob": getattr(seg, "avg_logprob", None),
            "no_speech_prob": getattr(seg, "no_speech_prob", None),
            "compression_ratio": getattr(seg, "compression_ratio", None),
        })
    text = "".join(parts).strip()
    return postprocess_transcript(text, meta)  # inert hook


# ==========================================================================
# Translation: DeepSeek
# ==========================================================================
async def translate_text(text: str, context_history: list, target_lang: str) -> str:
    if not text:
        return ""
    if deepseek_client is None:
        return "[translation unavailable: DEEPSEEK_API_KEY not set]"
    target_name = LANG_NAMES.get(target_lang, "中文")
    history_str = " ".join(context_history)
    system_prompt = f"Translate the sentence into {target_name}. Output only the translation. Context: {history_str}"
    try:
        resp = await deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Sentence: {text}"},
            ],
            timeout=15,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DeepSeek error] {e}")
        return f"[{target_name} translation failed]"


# ==========================================================================
# App lifecycle
# ==========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global deepseek_client
    if DEEPSEEK_API_KEY:
        deepseek_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        print("[*] DeepSeek client ready.")
    else:
        print("[!] DEEPSEEK_API_KEY is empty — transcription works, translation will be disabled.")
        print("    Copy .env.example to .env and fill in your key.")
    url = f"http://{HOST}:{PORT}/"
    print(f"[*] BroN-translate (Local) running at {url}")
    if AUTO_OPEN_BROWSER:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    yield
    # nothing to clean up


app = FastAPI(title="BroN-translate (Local Edition)", lifespan=lifespan)


@app.get("/")
async def root():
    return RedirectResponse(url="/static/translate.html")


# ==========================================================================
# WebSocket: the live translation pipeline
# ==========================================================================
@app.websocket("/ws/translate")
async def ws_translate(websocket: WebSocket):
    await websocket.accept()

    # Per-session config (client overrides via a "config" text message)
    session = {"source_lang": "auto", "target_lang": "zh", "live_preview": False}
    context = []  # last couple of finalized sentences, for translation context

    # VAD / segmentation state
    audio_buffer = bytearray()
    vol_history = []
    silence_frames = 0

    # Tunables (seconds / frames)
    MIN_SEC, MAX_SEC, SILENCE_FRAMES = 4.0, 12.0, 12
    MIN_BYTES = int(SAMPLE_RATE * SAMPLE_WIDTH * MIN_SEC)
    MAX_BYTES = int(SAMPLE_RATE * SAMPLE_WIDTH * MAX_SEC)

    # Vosk recognizer is created lazily, only if live preview is enabled
    vosk_rec = None

    async def flush_segment():
        """Send the buffered segment to Whisper -> DeepSeek -> client."""
        nonlocal audio_buffer
        if len(audio_buffer) < int(SAMPLE_RATE * SAMPLE_WIDTH * 1.0):
            return
        segment = bytes(audio_buffer)
        audio_buffer = bytearray()
        try:
            en = await asyncio.to_thread(transcribe_segment, segment, session["source_lang"])
        except Exception as e:
            print(f"[Whisper error] {e}")
            return
        if not en:
            return
        await websocket.send_json({"type": "en", "content": en})
        zh = await translate_text(en, context, session["target_lang"])
        await websocket.send_json({"type": "zh", "content": zh})
        context.append(en)
        if len(context) > 2:
            context.pop(0)

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=0.4)
            except asyncio.TimeoutError:
                # On a pause, flush whatever we have if it's long enough.
                if len(audio_buffer) >= MIN_BYTES:
                    await flush_segment()
                    vol_history.clear()
                    silence_frames = 0
                continue

            text_data = message.get("text")
            bytes_data = message.get("bytes")

            # ---- control messages ----
            if text_data:
                try:
                    cfg = json.loads(text_data)
                except Exception:
                    continue
                if cfg.get("type") == "config":
                    src = cfg.get("source_lang", "auto")
                    tgt = cfg.get("target_lang", "zh")
                    session["source_lang"] = src if src in ("auto", "en", "zh", "ja", "ru", "ms") else "auto"
                    session["target_lang"] = tgt if tgt in ("zh", "en", "ja", "ru") else "zh"
                    session["live_preview"] = bool(cfg.get("live_preview", False))
                    if session["live_preview"] and vosk_rec is None:
                        try:
                            from vosk import KaldiRecognizer
                            vosk_rec = KaldiRecognizer(get_vosk(), SAMPLE_RATE)
                        except Exception as e:
                            session["live_preview"] = False
                            await websocket.send_json({
                                "type": "error",
                                "content": f"Live preview unavailable: {e}",
                            })
                continue

            # ---- audio frames ----
            if bytes_data:
                audio_buffer.extend(bytes_data)

                # Live preview: feed Vosk for instant partial captions.
                if session["live_preview"] and vosk_rec is not None:
                    try:
                        if await asyncio.to_thread(vosk_rec.AcceptWaveform, bytes_data):
                            res = json.loads(vosk_rec.Result())
                            # ignore Vosk finals; Whisper produces the real text
                        else:
                            partial = json.loads(vosk_rec.PartialResult()).get("partial", "").strip()
                            if partial:
                                await websocket.send_json({"type": "partial", "content": partial})
                    except Exception:
                        pass

                # Simple energy-based VAD for segmentation.
                audio_np = np.frombuffer(bytes_data, dtype=np.int16)
                if audio_np.size:
                    volume = float(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)))
                    vol_history.append(volume)
                    if len(vol_history) > 25:
                        vol_history.pop(0)
                    recent_peak = max(vol_history) if vol_history else 300.0
                    threshold = max(250.0, recent_peak * 0.25)
                    silence_frames = silence_frames + 1 if volume < threshold else 0

                cur = len(audio_buffer)
                ended_on_silence = silence_frames >= SILENCE_FRAMES and cur >= MIN_BYTES
                too_long = cur >= MAX_BYTES
                if ended_on_silence or too_long:
                    await flush_segment()
                    vol_history.clear()
                    silence_frames = 0

    except WebSocketDisconnect:
        pass
    except RuntimeError:
        # client went away mid-receive; nothing to do
        pass
    except Exception as e:
        print(f"[ws error] {e}")


# Static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
