"""Synthesise the narration script (video/narration_ja.json) into video/build/narration/.

Two back ends, both local (the text is not sent to any web service):

    python tts.py voicevox [style id]   # VOICEVOX ENGINE running at 127.0.0.1:50021
                                        # default: 青山龍星 (low male voice), style ノーマル
    .venv-tts/bin/python tts.py kokoro [voice]   # Kokoro-82M (Apache-2.0, pip install kokoro "misaki[ja]")
                                                 # default: jm_kumo (male)

Writes one 24-kHz mono WAV per line (leading/trailing silence trimmed) and lines.json
(back end, credit line for the video, durations, and for VOICEVOX the kana reading of
every line, to check pronunciations). The narration text spells numbers and some terms
in kana because both voices read digit strings and a few kanji words wrongly.
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
import wave
from io import BytesIO
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "narration_ja.json"
OUT = HERE / "build" / "narration"
SR = 24000
KOKORO_SPEED = 1.2
VOICEVOX = "http://127.0.0.1:50021"
VV_SPEAKER = "青山龍星"
VV_PARAMS = dict(speedScale=1.05, pitchScale=0.0, intonationScale=1.1, volumeScale=1.0,
                 prePhonemeLength=0.05, postPhonemeLength=0.05)


def trim(a, margin=0.04):
    loud = np.flatnonzero(np.abs(a) > 0.02 * np.abs(a).max())
    if not len(loud):
        return a
    m = int(margin * SR)
    return a[max(0, loud[0] - m): loud[-1] + m]


def write_wav(path, a):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def vv(path, params=None, body=None):
    url = f"{VOICEVOX}{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
    data = json.dumps(body).encode() if body is not None else (b"" if params is not None else None)
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def voicevox_backend(args):
    speakers = json.loads(vv("/speakers"))
    if args:
        sid = int(args[0])
        name = next(s["name"] for s in speakers for st in s["styles"] if st["id"] == sid)
    else:
        sp = next(s for s in speakers if s["name"] == VV_SPEAKER)
        name, sid = sp["name"], next(st["id"] for st in sp["styles"] if st["name"] == "ノーマル")

    def say(text):
        q = json.loads(vv("/audio_query", dict(text=text, speaker=sid)))
        q.update(VV_PARAMS, outputSamplingRate=SR, outputStereo=False)
        with wave.open(BytesIO(vv("/synthesis", dict(speaker=sid), q))) as f:
            a = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768
        return a, q.get("kana", "")
    return say, dict(backend="voicevox", voice=f"{name} (style {sid})", credit=f"VOICEVOX:{name}")


def kokoro_backend(args):
    from kokoro import KPipeline
    voice = args[0] if args else "jm_kumo"
    pipe = KPipeline(lang_code="j", repo_id="hexgrad/Kokoro-82M")
    gap = np.zeros(int(0.12 * SR), np.float32)

    def say(text):
        parts, phon = [], []
        for _, ps, audio in pipe(text, voice=voice, speed=KOKORO_SPEED):
            parts += [np.asarray(audio, np.float32), gap]
            phon.append(ps)
        return np.concatenate(parts[:-1]), " | ".join(phon)
    return say, dict(backend="kokoro", voice=voice, credit=f"Kokoro-82M（{voice}）")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("voicevox", "kokoro"):
        sys.exit(__doc__)
    say, meta = (voicevox_backend if sys.argv[1] == "voicevox" else kokoro_backend)(sys.argv[2:])
    OUT.mkdir(parents=True, exist_ok=True)
    lines = json.loads(SCRIPT.read_text(encoding="utf-8"))
    done = []
    for x in lines:
        a, reading = say(x["text"])
        a = trim(a)
        write_wav(OUT / f"{x['id']}.wav", a)
        done.append(dict(id=x["id"], text=x["text"], dur=round(len(a) / SR, 3), reading=reading))
        print(f"{x['id']:<12} {len(a) / SR:5.1f} s  {reading if meta['backend'] == 'voicevox' else ''}", flush=True)
    (OUT / "lines.json").write_text(json.dumps(dict(meta, sr=SR, lines=done), ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print(meta["credit"], f"{sum(d['dur'] for d in done):.1f} s in total")


if __name__ == "__main__":
    main()
