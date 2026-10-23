#!/usr/bin/env python3
"""Selecciona los mejores clips por hablante desde transcripciones de ventanas de Whisper.

- Filtra por calidad (avg_logprob, no_speech_prob, compression_ratio, duración).
- Calcula prosodia por segmento: pitch (autocorrelación), energía RMS, palabras/segundo.
- Estima cuántas personas hablan: k-means 1D sobre log-pitch mediano, k ∈ 1..3 elegido por silhouette.
- Por hablante, elige clips con variedad (velocidad × entonación × intención) y los corta a WAV.

Uso:
  python3 select_clips.py --transcript-dir transcript_win --out-dir clips \
      [--max-speakers 3] [--max-per-speaker 40] [--min-dur 3] [--max-dur 15]

Requiere: ffmpeg, numpy. Salida: clips/<speaker>/clip_XXX.wav + .txt, metadata por hablante y summary.json.
"""
import argparse, csv, json, math, re, subprocess, wave
from pathlib import Path
import numpy as np

SR = 24000
PAD = 0.12


def load_audio(path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != SR:  # remuestreo lineal simple si el maestro no está a 24k
        a = np.interp(np.linspace(0, len(a), int(len(a) * SR / sr)), np.arange(len(a)), a).astype(np.float32)
    return a


def f0_stats(seg):
    fl, hop = int(0.04 * SR), int(0.02 * SR)
    mn, mx = int(SR / 500), int(SR / 75)
    f0 = []
    for i in range(0, len(seg) - fl, hop):
        fr = seg[i:i + fl]; fr = fr - fr.mean()
        if np.sqrt((fr ** 2).mean()) < 0.005:
            continue
        ac = np.correlate(fr, fr, "full")[len(fr) - 1:]
        ac[:mn] = 0
        if mx >= len(ac):
            continue
        lag = int(np.argmax(ac[:mx]))
        if lag > 0 and ac[lag] / (ac[0] + 1e-9) > 0.3:
            f0.append(SR / lag)
    return np.array(f0)


def intention(text):
    if re.search(r"[?¿]", text):
        return "pregunta"
    if re.search(r"[!¡]", text):
        return "exclamacion"
    return "narracion"


def kmeans1d(x, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.choice(x, size=k, replace=False)
    for _ in range(iters):
        d = np.abs(x[:, None] - c[None, :])
        lab = d.argmin(axis=1)
        nc = np.array([x[lab == j].mean() if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    return lab, c


def silhouette1d(x, lab, k):
    if k == 1:
        return 0.0
    s = []
    for i in range(len(x)):
        same = x[(lab == lab[i]) & (np.arange(len(x)) != i)]
        a = np.abs(same - x[i]).mean() if len(same) else 0.0
        b = min(np.abs(x[lab == j] - x[i]).mean() for j in range(k) if j != lab[i])
        s.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-speakers", type=int, default=3)
    ap.add_argument("--max-per-speaker", type=int, default=40)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=15.0)
    args = ap.parse_args()

    tdir = Path(args.transcript_dir)
    manifest = json.loads((tdir / "manifest.json").read_text())
    audio = load_audio(manifest["master"])
    starts = manifest["starts"]

    segs = []
    for jf in sorted(tdir.glob("win_*.json")):
        off = float(starts[jf.stem])
        for s in json.loads(jf.read_text(encoding="utf-8"))["segments"]:
            st, en = s["start"] + off, s["end"] + off
            dur = en - st
            text = re.sub(r"\s+", " ", s["text"]).strip()
            if not (args.min_dur <= dur <= args.max_dur) or len(text) < 12:
                continue
            if re.search(r"(.)\1{6,}", text) or "♪" in text:
                continue
            if s.get("avg_logprob", -9) < -0.55 or s.get("no_speech_prob", 1) > 0.35 or s.get("compression_ratio", 99) > 1.8:
                continue
            f0 = f0_stats(audio[int(st * SR):int(en * SR)])
            if len(f0) < 5:
                continue
            rms = float(np.sqrt((audio[int(st * SR):int(en * SR)] ** 2).mean()))
            peak = float(np.abs(audio[int(st * SR):int(en * SR)]).max())
            if peak > 0.99:
                continue
            segs.append({"start": st, "end": en, "dur": round(dur, 3), "text": text,
                         "pitch_mean": float(np.median(f0)),
                         "pitch_range": float(np.percentile(f0, 90) - np.percentile(f0, 10)),
                         "rms": rms, "wps": round(len(text.split()) / dur, 2),
                         "avg_logprob": round(s["avg_logprob"], 4),
                         "intencion": intention(text)})
    print(f"Candidatos con prosodia: {len(segs)}")
    if not segs:
        return

    # --- Estimar número de hablantes por pitch mediano ---
    x = np.log2(np.array([s["pitch_mean"] for s in segs]))
    best = (1, np.zeros(len(x), dtype=int), 0.0)
    for k in range(1, args.max_speakers + 1):
        lab, _ = kmeans1d(x, k)
        sil = silhouette1d(x, lab, k)
        print(f"k={k}: silhouette={sil:.3f}")
        if sil > best[2] + 0.02:  # exigir mejora mínima para añadir un hablante
            best = (k, lab, sil)
    k, labels, sil = best
    print(f"Hablantes estimados: {k} (silhouette {sil:.2f}; valores <0.35 = incierto, confirmar escuchando)")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    summary = {"speakers_estimated": k, "silhouette": round(sil, 3), "speakers": {}}
    order = np.argsort([segs[i]["pitch_mean"] for i in range(len(segs))])
    centers = sorted(((np.mean([segs[i]["pitch_mean"] for i in range(len(segs)) if labels[i] == j]), j)
                      for j in range(k)))
    name_of = {j: f"speaker_{n + 1}" for n, (_, j) in enumerate(centers)}

    for j in range(k):
        name = name_of[j]
        group = [s for i, s in enumerate(segs) if labels[i] == j]
        pitches = [s["pitch_mean"] for s in group]
        wps = np.array([s["wps"] for s in group]); pv = np.array([s["pitch_range"] for s in group]); rm = np.array([s["rms"] for s in group])
        q1w, q3w = np.percentile(wps, [25, 75]); q1p, q3p = np.percentile(pv, [25, 75]); q3r = np.percentile(rm, 75)
        for s in group:
            tags = [s["intencion"],
                    "rapido" if s["wps"] >= q3w else "lento" if s["wps"] <= q1w else "ritmo_medio",
                    "entonacion_alta" if s["pitch_range"] >= q3p else "entonacion_plana" if s["pitch_range"] <= q1p else "entonacion_media"]
            if s["rms"] >= q3r:
                tags.append("energico")
            s["tags"] = ",".join(tags)
        buckets = {}
        for s in sorted(group, key=lambda z: -z["avg_logprob"]):
            key = (s["intencion"], s["tags"].split(",")[1], s["tags"].split(",")[2])
            buckets.setdefault(key, []).append(s)
        sel = []
        for _, lst in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            sel.extend(lst[:3])
        sel = sel[:args.max_per_speaker]

        sdir = out / name; sdir.mkdir(exist_ok=True)
        rows = []
        for idx, s in enumerate(sel):
            fn = f"clip_{idx:03d}"
            st = max(0.0, s["start"] - PAD)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", manifest["master"],
                            "-ss", f"{st:.3f}", "-t", f"{(s['end'] + PAD) - st:.3f}",
                            "-c:a", "pcm_s16le", str(sdir / f"{fn}.wav")], check=True)
            (sdir / f"{fn}.txt").write_text(s["text"] + "\n", encoding="utf-8")
            rows.append({"file": f"{fn}.wav", "start": round(s["start"], 2), "dur": s["dur"],
                         "wps": s["wps"], "pitch_mean": round(s["pitch_mean"], 1),
                         "pitch_range": round(s["pitch_range"], 1), "rms": round(s["rms"], 4),
                         "avg_logprob": s["avg_logprob"], "tags": s["tags"], "text": s["text"]})
        with open(sdir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        summary["speakers"][name] = {"candidates": len(group), "clips": len(rows),
                                     "pitch_median_hz": round(float(np.median(pitches)), 1),
                                     "minutes_selected": round(sum(r["dur"] for r in rows) / 60, 2)}
        print(f"{name}: {len(group)} candidatos → {len(rows)} clips · pitch mediano {np.median(pitches):.0f} Hz")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Resumen:", out / "summary.json")


if __name__ == "__main__":
    main()
