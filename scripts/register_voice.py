#!/usr/bin/env python3
"""Registra una voz en el servicio local Qwen3-TTS (openmontage local_tts) y genera un audio de prueba.

Uso:
  python3 register_voice.py --voice-id nina --name "Nina Cervantes" \
      --clip referencias/nina.wav --transcript referencias/nina.txt \
      --test-text "Hola, esta es una prueba." --out generados/prueba_nina.wav

Requiere solo la librería estándar. El servicio debe estar corriendo (python -m local_tts, puerto 8765).
"""
import argparse, json, sys, time, urllib.request, urllib.error
from pathlib import Path


def post(base, path, data, timeout=60):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", default="http://127.0.0.1:8765")
    ap.add_argument("--voice-id", required=True, help="[A-Za-z0-9_-]{1,80}")
    ap.add_argument("--name", required=True)
    ap.add_argument("--clip", required=True, help="WAV de referencia (1–60 s; ideal 5–15 s)")
    ap.add_argument("--transcript", required=True, help="archivo .txt con la transcripción exacta del clip")
    ap.add_argument("--language", default="es", help="en es fr de it pt ru zh ja ko")
    ap.add_argument("--description", default="")
    ap.add_argument("--test-text", default=None, help="si se da, crea un job y lo ensambla en --out")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wait", type=int, default=600, help="segundos máximos de espera del job")
    args = ap.parse_args()

    try:
        h = get(args.service, "/health")
    except (urllib.error.URLError, OSError):
        sys.exit(f"Servicio no disponible en {args.service}. Inícialo desde openmontage con: .venv/bin/python -m local_tts")
    if not h.get("models", {}).get("Base-8bit"):
        sys.exit("El modelo Base-8bit no está descargado en el servicio (ver QWEN3_DATA_DIR/models).")

    clip = Path(args.clip).expanduser().resolve()
    text = Path(args.transcript).read_text(encoding="utf-8").strip()
    profile = post(args.service, "/voices", {
        "voice_id": args.voice_id, "name": args.name, "language_code": args.language,
        "reference_path": str(clip), "reference_text": text, "description": args.description})
    dur = profile["references"][args.language]["duration_seconds"]
    print(f"Voz '{args.voice_id}' registrada ({dur:.1f}s de referencia).")

    if not args.test_text:
        return
    if not args.out:
        sys.exit("--out es obligatorio cuando se usa --test-text")
    job = post(args.service, "/jobs", {"voice_id": args.voice_id, "language_code": args.language,
                                       "text": args.test_text, "auto_approve": True})
    jid = job["job_id"]
    print(f"Job {jid} ({len(job['segments'])} segmentos), generando…", flush=True)
    deadline = time.time() + args.wait
    while time.time() < deadline:
        st = get(args.service, f"/jobs/{jid}")
        if st["status"] == "completed":
            break
        if st["status"] in ("failed", "cancelled"):
            sys.exit(f"Job {st['status']}: {st.get('error')}")
        time.sleep(6)
    else:
        sys.exit("Tiempo de espera agotado; el job sigue en el servicio.")
    out = post(args.service, f"/jobs/{jid}/assemble", {"output_path": str(Path(args.out).expanduser().resolve())})
    print(f"Audio de prueba: {out['artifacts'][0]} ({out['duration_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
