# API del servicio local Qwen3-TTS (openmontage `local_tts`)

Servicio FastAPI en `http://127.0.0.1:8765`. Arranque (desde el repo openmontage):

```bash
.venv/bin/python -m local_tts   # solo loopback; usa QWEN3_DATA_DIR o ~/.local/share/openmontage/qwen3
```

Datos en `~/.local/share/openmontage/qwen3/`: `voices/<voice_id>/profile.json` + WAVs de referencia,
`jobs/<job_id>/job.json`, `cache/` (segmentos por hash), `models/{Base-8bit,VoiceDesign-8bit}`.

## Endpoints

| Método | Ruta | Cuerpo / notas |
|---|---|---|
| GET | `/health` | `models.Base-8bit` debe ser `true` para clonar |
| GET | `/voices` | lista perfiles registrados |
| POST | `/voices` | `{voice_id, name?, language_code, reference_path, reference_text, description?}` — registra/actualiza; `create_only: true` falla si el id existe. Referencia: 1–60 s, recomendado 5–15 s, transcripción EXACTA obligatoria |
| POST | `/jobs` | `{voice_id, language_code, text, auto_approve: true, settings?}` — segmenta el texto solo; con `auto_approve` termina en `completed` |
| GET | `/jobs/{job_id}` | `status`: queued → running → completed / awaiting_review / failed |
| POST | `/jobs/{job_id}/assemble` | `{output_path: "<abs>.wav", mp3?, silence_seconds?}` — une segmentos aprobados |
| POST | `/jobs/{job_id}/cancel` · `/resume` · `/approve` | control de jobs |
| GET | `/jobs/{job_id}/segments/{id}/audio` | WAV del segmento |
| POST | `/design` | `{language_code, text, instructions}` — voz nueva desde descripción (modelo VoiceDesign-8bit) |

Idiomas (`language_code`): `en es fr de it pt ru zh ja ko`.

## Notas

- Un solo worker: los jobs se serializan; lanzar varios y hacer polling está bien.
- `settings` admite `temperature, top_p, top_k, repetition_penalty, max_tokens` (positivos; `top_p ≤ 1`; `max_tokens` y `top_k` enteros). Default `max_tokens=2048`.
- Errores 400 con mensaje en español: referencia >60 s, silencio, hash de referencia cambiado, etc.
- Voces ya registradas típicas en esta máquina: consultar siempre `GET /voices` antes de crear ids nuevos para no duplicar.
