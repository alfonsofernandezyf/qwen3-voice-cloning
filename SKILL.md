---
name: qwen3-voice-cloning
description: Pipeline para clonar voces con el servicio local Qwen3-TTS de openmontage (127.0.0.1:8765) a partir de un video de YouTube o una grabación local (podcast, entrevista, video): descargar el audio, transcribir ventanas con Whisper, estimar cuántas personas hablan, seleccionar los mejores clips por hablante con variedad de entonación/velocidad/intención, registrar cada voz en el servicio y generar audios de prueba. Usar cuando el usuario pida clonar/extraer/sacar voces de un video o grabación, crear voces nuevas para Qwen3-TTS, preparar datasets de voz (clips WAV + transcripción) para modelos de voz, o registrar/actualizar voces en el servicio local de TTS.
---

# Qwen3 Voice Cloning Pipeline

Pipeline de 6 pasos. Todo el trabajo dentro del workspace actual. Scripts en `scripts/` (rutas relativas a este SKILL.md).

Prerrequisitos: `ffmpeg` y Whisper CLI (openai-whisper, homebrew). Para YouTube: `yt-dlp` (si falta: `brew install yt-dlp`). El servicio local_tts debe estar corriendo: `curl -s http://127.0.0.1:8765/health` debe dar `"ready": true`; si no, arrancarlo con `nohup` desde `~/CODE/openmontage`: `.venv/bin/python -m local_tts`.

## 1. Obtener el audio

- YouTube: `yt-dlp -x --audio-format wav -o "fuente.%(ext)s" "<URL>"` (pedir la URL si el usuario no la dio).
- Archivo local: copiarlo al workspace (nunca trabajar sobre el original).
- Extraer maestro mono 24 kHz (estándar Qwen3-TTS): `ffmpeg -i fuente.* -ac 1 -ar 24000 master_24k.wav`. Verificar duración con `ffprobe` (errores de paquete Opus al final son ignorables si la duración cuadra).

## 2. Transcribir ventanas (no transcribir todo salvo que se pida)

Muestrear ~16 ventanas de 90 s repartidas uniformemente. El módulo whisper solo es importable por el intérprete del CLI de homebrew; ejecutar así (una sola carga del modelo, ~30 s por ventana de 90 s en Apple Silicon):

```bash
PYW=$(head -1 "$(command -v whisper)" | tr -d '#!')
"$PYW" scripts/transcribe_windows.py --master master_24k.wav \
  --out-dir transcript_win --windows-dir windows --n 16 --win-sec 90 --language es
```

Para grabaciones >2 h o muchas ventanas, lanzar con `nohup` y hacer polling del log. El script reanuda: relanzar continúa donde quedó.

## 3. Estimar hablantes y seleccionar clips

```bash
python3 scripts/select_clips.py --transcript-dir transcript_win --out-dir clips
```

- Imprime silhouette por k y el número estimado de personas. **Siempre reportar la estimación al usuario y confirmar** (silhouette <0.35 = incierto); la separación es por pitch (heurística, confunde voces del mismo registro). Opción robusta si se requiere certeza: diarización con pyannote (instalación aparte).
- **Dos voces del mismo registro (p. ej. dos hombres, entrevistador + invitado): el pitch NO las separa** (k-means mezcla clusters aunque el silhouette parezca decente). Recurso efectivo: curaduría por contenido — filtrar segmentos por marcadores de primera persona del hablante objetivo (p. ej. artista: "mi música/mi disco/el show/escenario/gira"; excluir marcadores del entrevistador: "mano/bro/tu música/la pregunta/el podcast"), luego verificar coherencia acústica del set (descartar pitch mediano fuera de ±33 % de la mediana) y construir la compuesta solo con esos segmentos. Pistas adicionales: buscar menciones del nombre ("te lo está diciendo X", "aquí está X") y usar la respuesta siguiente como ancla; el usuario suele saber en qué minuto habla quién — pedir/aprovechar ese dato y transcribir la intro aparte para perfilar al entrevistador.
- **Vocativo inverso (ancla más confiable en entrevistas): quien PRONUNCIA el nombre no es quien LO TIENE.** El invitado dice "Igor, te digo…" y el entrevistador dice "Lula/presidente/senhor, …". Clasificar con regex de vocativos + rango de pitch mediano como segundo eje (p. ej. invitado ≥160 Hz, host <175 Hz), luego expandir con segmentos sin vocativo dentro del rango. Ojo con la 3ª persona reflexiva ("o Lula não gasta demais" dicho por el propio Lula): el vocativo manda, la mención sola no. Cuando un hablante domina (invitado ~85 %), su voz se parte en varios clusters de pitch por expresividad — unirlos por anclas, no por k-means. Revisar SIEMPRE la lista final segmento por segmento: el pitch bajo contamina (pasajes calmados del invitado caen en el rango del host); descartar los que suenen al otro hablante aunque el pitch encaje.
- Salida: `clips/speaker_N/clip_XXX.wav + .txt` (3–15 s, 24 kHz mono), `metadata.csv` por hablante (wps, pitch, rms, tags de variedad) y `clips/summary.json`.
- Pedir al usuario que identifique cada `speaker_N` (nombre real) antes de registrar.

## 4. Registrar voces en el servicio

Detalles del API en `references/service-api.md` (leer si un endpoint falla). Antes de registrar, `GET /voices` para no duplicar ids existentes. El servicio guarda UNA referencia por idioma (1–60 s); referencias largas y variadas clonan mucho mejor.

**Fuentes conversacionales (podcasts, entrevistas, charlas): los clips cortos casi siempre contienen interrupciones del otro hablante y degradan el clon.** Construir una referencia compuesta de 45–55 s solo con segmentos limpios (verificado frame a frame por pitch) y expresividad variada:

```bash
python3 scripts/build_composite.py --transcript-dir transcript_win --voice martha \
  --pitch-min 185 --max-low 0.08 --target-sec 52 --out referencias/referencia_martha
# voz grave: --pitch-max 175 --max-high 0.15
```

Ajustar `--pitch-min/--pitch-max` según el `pitch_median_hz` de `clips/summary.json`. Para fuentes de UN solo hablante (locución, audiolibro) basta un clip limpio de 5–15 s de `clips/speaker_1/` o una grabación directa.

```bash
python3 scripts/register_voice.py --voice-id <id> --name "<Nombre>" \
  --clip referencias/referencia_martha.wav --transcript referencias/referencia_martha.txt \
  --test-text "<frase de prueba en el idioma>" --out generados/prueba_<id>.wav
```

Si el usuario reporta que el clon "no suena parecido", casi siempre es contaminación o referencia demasiado corta/plana: regenerar la compuesta con umbrales más estrictos (`--max-low 0.05`) o más duración.

**Descripción de la voz (`--description`): siempre neutra — solo rasgos de la voz** (registro, timbre, ritmo, estilo, p. ej. "Voz masculina grave y relajada; conversacional."). NUNCA mencionar la fuente, el programa, la URL, ni que fue extraída de un video/podcast/entrevista.

**Proveniencia (regla general): no dejar rastros de la fuente en NINGÚN artefacto.** Ni descripciones, ni nombres de carpetas/archivos, ni metadata.csv/README, ni los TEXTOS de referencia deben mencionar el programa, canal, marca o URL de origen. Al construir compuestas, excluir además los segmentos que nombren el programa/canal/marca (intros tipo "bienvenidos a…", "seguimos en…", saludos a la conductora por su nombre). Los JSON de transcripción de trabajo se quedan como material bruto; lo que se registra en el servicio debe ir limpio.

## 5. Validar y entregar

- Verificar el WAV generado (duración >0, RMS ~0.05–0.2, peak <0.95) con `wave`+`numpy` o `ffprobe`.
- Entregar: rutas de los audios de prueba, ids de voz registrados, estructura de `clips/` y advertir si la separación de hablantes fue incierta.

## 6. Notas

- Mezclar hablantes en una sola voz arruina la clonación: una voz = una persona.
- Grabación con música/ruido de fondo constante degrada el clon; preferir clips con `avg_logprob` alto.
- Si el usuario quiere TODO el audio transcrito: `--n` alto o transcribir `master_24k.wav` directo con el mismo intérprete y `whisper.load_model(...).transcribe(...)`; ~0.5× tiempo real en M-series.
- El texto a generar puede ser largo: el servicio segmenta solo; ensamblar con `auto_approve: true` para uso no interactivo.
