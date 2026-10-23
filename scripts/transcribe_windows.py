#!/usr/bin/env python3
"""Extrae ventanas uniformes de un audio maestro y las transcribe con Whisper (modelo cargado una sola vez).

Uso:
  python3 transcribe_windows.py --master master_24k.wav --out-dir transcript_win \
      --windows-dir windows --n 16 --win-sec 90 --model large-v3-turbo --language es

Requiere: ffmpeg en PATH y el módulo `whisper` (openai-whisper) importable por el intérprete usado.
Escribe windows/win_XX.wav, transcript_win/win_XX.json y transcript_win/manifest.json (offsets en segundos).
"""
import argparse, json, subprocess, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--win-sec", type=float, default=90)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--language", default="es")
    ap.add_argument("--margin", type=float, default=60, help="segundos a omitir al inicio y al final")
    args = ap.parse_args()

    master = Path(args.master)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    win_dir = Path(args.windows_dir); win_dir.mkdir(parents=True, exist_ok=True)

    dur = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(master)]).strip())
    span = dur - 2 * args.margin - args.win_sec
    if span <= 0:
        sys.exit(f"Audio demasiado corto ({dur:.0f}s) para {args.n} ventanas de {args.win_sec}s con margen {args.margin}s")
    starts = [args.margin + i * span / (args.n - 1) for i in range(args.n)] if args.n > 1 else [args.margin]

    done = {p.stem for p in out_dir.glob("win_*.json")}
    todo = [(i, s) for i, s in enumerate(starts) if f"win_{i:02d}" not in done]
    print(f"Duración {dur:.0f}s · {args.n} ventanas · pendientes: {len(todo)}", flush=True)

    for i, s in enumerate(starts):
        w = win_dir / f"win_{i:02d}.wav"
        if not w.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.2f}", "-t", str(args.win_sec),
                            "-i", str(master), "-ac", "1", "-ar", "16000", str(w)], check=True)

    if todo:
        import whisper
        model = whisper.load_model(args.model)
        for i, s in todo:
            r = model.transcribe(str(win_dir / f"win_{i:02d}.wav"), language=args.language, fp16=False, verbose=False)
            (out_dir / f"win_{i:02d}.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
            print(f"OK win_{i:02d} ({len(r['segments'])} segmentos)", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(
        {"master": str(master.resolve()), "duration": dur, "win_sec": args.win_sec,
         "starts": {f"win_{i:02d}": round(s, 3) for i, s in enumerate(starts)}}, indent=2))
    print("Listo. Manifest:", out_dir / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
