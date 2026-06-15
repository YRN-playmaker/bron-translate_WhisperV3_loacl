# BroN-translate · Local Edition

**Self-hosted, real-time speech translation that runs entirely on your own machine.**

This is the local, open-source build of [BroN-translate](https://bron-translate.org):
a single-process server that captures microphone audio, transcribes it with a
local **Whisper large-v3** model, translates with **DeepSeek**, and shows live
captions in your browser — including a draggable floating-window (PiP) overlay
for use during lectures, meetings, or streams.

No accounts. No usage metering. No cloud gateway. You bring your own GPU and your
own DeepSeek API key, and everything else stays on `127.0.0.1`.

> Looking for a zero-setup, nothing-to-install version? Use the hosted service at
> **[bron-translate.org](https://bron-translate.org)** instead. This repository is
> for people who want to run the model locally on their own hardware.

---

## Pipeline

```
microphone (PCM)
   │
   ├─ (optional) Vosk  ──► instant word-by-word "live preview" captions
   │
   ▼
energy-based VAD  ──► splits speech into segments
   │
   ▼
faster-whisper (large-v3)  ──► final transcription of each segment
   │
   ▼
DeepSeek  ──► translation
   │
   ▼
captions over WebSocket  ──► browser + floating PiP window
```

Two layers of recognition: **Vosk** (optional, toggled in the UI) gives instant
interim text while you speak, and **Whisper large-v3** produces the accurate
final transcription that gets translated. Turn live preview off and it's pure
Whisper.

---

## Requirements

- **Python 3.10+**
- An **NVIDIA GPU** is strongly recommended for Whisper large-v3. CPU works but is
  slow for real-time use. (faster-whisper handles the CUDA/cuDNN runtime via its
  own wheels.)
- A **DeepSeek API key** for translation — [platform.deepseek.com](https://platform.deepseek.com).
  Transcription runs without it; translation won't until you add it.

---

## Quick start

### Windows
1. Install Python 3.10+ (check *"Add Python to PATH"*).
2. Double-click **`start.bat`**. The first run creates a virtual environment,
   installs dependencies, and opens `.env` for you to paste your DeepSeek key.
3. Run `start.bat` again — your browser opens to the app.

### macOS / Linux
```bash
git clone https://github.com/YRN-playmaker/bron-translate-local.git
cd bron-translate-local
./start.sh            # first run: sets up venv + deps, then creates .env
# edit .env, paste your DEEPSEEK_API_KEY
./start.sh            # starts the server and opens your browser
```

### Manual (any OS)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # then edit .env
python main.py
```

On first transcription, faster-whisper downloads the chosen model weights
automatically (large-v3 is a few GB). Subsequent runs are instant.

> **RTX 50-series (Blackwell) users / 中文用户:** there's a detailed
> step-by-step setup guide — including the cuBLAS/cuDNN version pinning and the
> Windows DLL fix needed for Blackwell cards — in [`读我.md`](读我.md).

---

## Configuration

Everything lives in `.env` (copy from `.env.example`):

| Variable | Default | Notes |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | required for translation |
| `WHISPER_MODEL` | `large-v3` | full-accuracy default; `large-v3-turbo` is faster but less accurate |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `WHISPER_COMPUTE_TYPE` | `float16` | `int8_float16` / `int8` for lower VRAM. **RTX 50-series (Blackwell): use `float16` or `float32` only — `int8` is currently broken on these cards.** |
| `VOSK_MODEL_PATH` | `model/vosk-model-small-en-us` | only needed if you enable live preview |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | local server address |
| `AUTO_OPEN_BROWSER` | `1` | open the UI on launch |

### Optional: live preview (Vosk)
Download a model from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
(e.g. `vosk-model-small-en-us-0.15`), unzip it into `model/`, and point
`VOSK_MODEL_PATH` at it. Then flip the **⚡ live preview** switch in the UI.

---

## Usage

1. Pick your microphone (or a virtual audio cable to capture system sound).
2. Choose source → target language.
3. Hit **▶ start**. Captions appear live.
4. Click **📺 floating** for a draggable always-on-top subtitle window.

---

## Notes for developers

The transcription path exposes two no-op extension points
(`refine_segment_audio` and `postprocess_transcript` in `main.py`) so the
pipeline can be extended without touching the surrounding code. They are inert
pass-throughs in this edition.

---

## License & Commercial Use

This project is licensed under the **GNU Affero General Public License v3.0
(AGPL-3.0)**. See [`LICENSE`](LICENSE) for the full text.

In plain terms:

- You are free to **use, study, modify, and share** this software.
- If you modify it and **make it available to others over a network** (for
  example, by hosting it as a web service), the AGPL requires you to **release
  your complete corresponding source code** under the same AGPL-3.0 license.
- This network-use clause is what separates AGPL from ordinary GPL: running a
  modified version as a service counts as distribution.

**Commercial / closed-source licensing.** If you want to use this software in a
proprietary or commercial product **without** the AGPL's source-disclosure
obligations, a separate commercial license is available. Contact
**your-gmail@gmail.com** to arrange one.

Copyright © 2026 YRN-playmaker.

---

## Disclaimer

Translations are for everyday study and communication. Don't rely on them for
legal, medical, or otherwise critical use.
