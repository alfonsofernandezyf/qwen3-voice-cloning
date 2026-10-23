#!/usr/bin/env python3
"""Construye una referencia compuesta (hasta 60 s) de UN solo hablante, sin interrupciones.

Recorre TODOS los segmentos transcritos, mide el pitch frame a frame y descarta los que
tienen voz del otro hablante (contaminación). Selecciona por expresividad (rango de pitch)
y concatena con pausas de 0.35 s; genera el WAV y la transcripción exacta para registrarla.

Uso típico (dos hablantes):
  python3 build_composite.py --transcript-dir transcript_win --voice martha \
      --pitch-min 185 --max-low 0.08 --target-sec 52 --out referencias/referencia_martha_v2
  python3 build_composite.py --transcript-dir transcript_win --voice yordi \
      --pitch-max 175 --max-high 0.15 --target-sec 45 --out referencias/referencia_yordi_v2

--pitch-min/--pitch-max: rango del pitch mediano del hablante (Hz). --max-low: fracción
máxima de frames <165 Hz (voz grave intrusa); --max-high: fracción máxima >200 Hz (aguda).
"""
import argparse, json, re, wave
from pathlib import Path
import numpy as np

SR = 24000


def f0_frames(audio, a, b):
    seg = audio[int(a * SR):int(b * SR)]
    fl, hop = int(0.04 * SR), int(0.02 * SR)
    mn, mx = int(SR / 500), int(SR / 75)
    out = []
    for i in range(0, len(seg) - fl, hop):
        fr = seg[i:i + fl]; fr = fr - fr.mean()
        if np.sqrt((fr ** 2).mean()) < 0.005:
            continue
        ac = np.correlate(fr, fr, "full")[len(fr) - 1:]
        ac[:mn] = 0
        lag = int(np.argmax(ac[:mx]))
        if lag > 0 and ac[lag] / (ac[0] + 1e-9) > 0.3:
            out.append(SR / lag)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript-dir", required=True)
    ap.add_argument("--voice", required=True, help="etiqueta solo para mensajes")
    ap.add_argument("--pitch-min", type=float, default=0)
    ap.add_argument("--pitch-max", type=float, default=1e9)
    ap.add_argument("--max-low", type=float, default=0.08, help="fracción máx. frames <165 Hz")
    ap.add_argument("--max-high", type=float, default=0.15, help="fracción máx. frames >200 Hz")
    ap.add_argument("--target-sec", type=float, default=50, help="duración objetivo (máx. 58)")
    ap.add_argument("--out", required=True, help="prefijo de salida (.wav/.txt)")
    args = ap.parse_args()

    man = json.loads((Path(args.transcript_dir) / "manifest.json").read_text())
    with wave.open(man["master"], "rb") as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    pool = []
    for jf in sorted(Path(args.transcript_dir).glob("win_*.json")):
        off = float(man["starts"][jf.stem])
        for s in json.loads(jf.read_text(encoding="utf-8"))["segments"]:
            st, en = s["start"] + off, s["end"] + off
            d = en - st
            t = re.sub(r"\s+", " ", s["text"]).strip()
            if not (2.5 <= d <= 15) or len(t) < 12:
                continue
            if s.get("avg_logprob", -9) < -0.6 or s.get("no_speech_prob", 1) > 0.4:
                continue
            f = f0_frames(audio, st, en)
            if len(f) < 8:
                continue
            if not (args.pitch_min <= float(np.median(f)) <= args.pitch_max):
                continue
            if float((f < 165).mean()) > args.max_low or float((f > 200).mean()) > args.max_high:
                continue
            pool.append({"st": st, "en": en, "dur": d, "text": t,
                         "range": float(np.percentile(f, 90) - np.percentile(f, 10)),
                         "q": s["avg_logprob"]})
    pool.sort(key=lambda s: (-s["range"], -s["q"]))  # expresividad primero, calidad después
    target = min(args.target_sec, 58)
    sel, tot = [], 0.0
    for s in pool:
        if tot + s["dur"] > target:
            continue
        sel.append(s); tot += s["dur"]
        if tot >= target * 0.9:
            break
    print(f"{args.voice}: {len(pool)} segmentos limpios disponibles → {len(sel)} usados ({tot:.1f}s)")

    sil = np.zeros(int(0.35 * SR), dtype=np.float32)
    parts, texts = [], []
    for s in sel:
        parts += [audio[int(s["st"] * SR):int(s["en"] * SR)], sil]
        t = s["text"].strip()
        if t and t[-1] not in ".!?…":
            t += "."
        texts.append(t)
    comp = np.concatenate(parts[:-1])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out) + ".wav", "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((comp * 32768).astype(np.int16).tobytes())
    Path(str(out) + ".txt").write_text(" ".join(texts) + "\n", encoding="utf-8")
    print(f"Escrito: {out}.wav ({len(comp)/SR:.1f}s) + .txt")


if __name__ == "__main__":
    main()
