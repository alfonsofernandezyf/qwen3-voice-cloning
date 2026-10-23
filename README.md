# qwen3-voice-cloning

Skill para agentes (Kimi, Claude Code, Cursor, etc.) que implementa un pipeline completo de **clonado de voz con Qwen3-TTS**: desde un video de YouTube o una grabación local (podcast, entrevista), descarga el audio, transcribe ventanas con Whisper, estima cuántas personas hablan, separa las voces, selecciona los mejores clips por hablante (con variedad de entonación, velocidad e intención), registra cada voz en un servicio local de TTS y genera audios de prueba.

Todo corre **100% en local**: la voz nunca sale de tu máquina.

## Qué incluye

```
SKILL.md                      # Instrucciones del pipeline para el agente (6 pasos)
scripts/
  transcribe_windows.py       # Muestreo de ventanas + transcripción Whisper (reanudable)
  select_clips.py             # Prosodia + clustering: estima hablantes y selecciona clips
  build_composite.py          # Referencia compuesta 45–55 s por rango de pitch
  register_voice.py           # Registro de voz + prueba vía API HTTP del servicio
references/
  service-api.md              # Contrato de la API del servicio local (endpoints, idiomas)
```

## Requisitos

- **macOS / Linux** (probado en Apple Silicon)
- `ffmpeg` (`brew install ffmpeg`)
- `yt-dlp` (`brew install yt-dlp`) — solo para fuentes de YouTube
- Whisper CLI (`pip install openai-whisper`, modelo `large-v3-turbo` recomendado)
- Un **servicio local de Qwen3-TTS** escuchando en `http://127.0.0.1:8765` (ver abajo)

## Instalación de Qwen3-TTS

Hay dos rutas. La recomendada en Mac es la A; la B es la oficial (GPU con CUDA).

### Opción A — openmontage `local_tts` (Apple Silicon, offline)

[OpenMontage](https://github.com/calesthio/OpenMontage) incluye un servicio local de Qwen3-TTS optimizado para Apple Silicon (modelos MLX 8-bit), con API HTTP, interfaz web y persistencia de voces/jobs. Es la ruta con la que este skill está probado.

```bash
git clone https://github.com/calesthio/OpenMontage.git
cd OpenMontage

# Entorno dedicado (uv + ffmpeg requeridos; uv provee Python 3.12)
scripts/setup_qwen3_tts.sh --download

# Descargar los modelos (una sola vez, con conexión)
~/.local/share/openmontage/qwen3/venv/bin/hf download \
  mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
  --local-dir ~/.local/share/openmontage/qwen3/models/Base-8bit
~/.local/share/openmontage/qwen3/venv/bin/hf download \
  mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit \
  --local-dir ~/.local/share/openmontage/qwen3/models/VoiceDesign-8bit

# Arrancar el servicio (loopback 127.0.0.1:8765)
scripts/start_qwen3_tts.sh
```

Verifica con `curl -s http://127.0.0.1:8765/health` → debe reportar `ready: true` y el modelo `Base-8bit` presente (es el de clonado; `VoiceDesign-8bit` es opcional, para diseñar voces desde una descripción).

- UI web: `http://127.0.0.1:8765/ui` · Docs de la API: `http://127.0.0.1:8765/docs`
- Datos (voces, jobs, modelos): `~/.local/share/openmontage/qwen3/` (configurable con `QWEN3_DATA_DIR`)
- **Sin autenticación**: mantén el binding en loopback. Las referencias de voz y los jobs contienen audio/texto privado.
- La inferencia fuerza modo offline de Hugging Face y solo carga directorios locales.

### Opción B — Qwen3-TTS oficial (QwenLM, GPU con CUDA)

Repositorio: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts
# recomendado: pip install -U flash-attn --no-build-isolation
```

Modelos: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (clonado), `-VoiceDesign` (diseño de voz), `-CustomVoice` (voces predefinidas). Demo web: `qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 8000`.

> **Nota de compatibilidad**: los scripts `transcribe_windows.py`, `select_clips.py` y `build_composite.py` son agnósticos al servicio (producen WAVs + transcripciones). Solo `register_voice.py` habla con la API HTTP del servicio de la Opción A; si usas la Opción B, registra las voces llamando a `generate_voice_clone` / `create_voice_clone_prompt` de `qwen_tts` directamente con los clips generados.

## Uso del skill con openmontage

Una vez corriendo el servicio (Opción A), el pipeline es:

```bash
# 1. Audio maestro mono 24 kHz
yt-dlp -x --audio-format wav -o "fuente.%(ext)s" "<URL>"
ffmpeg -i fuente.wav -ac 1 -ar 24000 master_24k.wav

# 2. Transcribir ~16 ventanas de 90 s (una carga de modelo, reanudable)
PYW=$(head -1 "$(command -v whisper)" | tr -d '#!')
"$PYW" scripts/transcribe_windows.py --master master_24k.wav \
  --out-dir transcript_win --windows-dir windows --n 16 --win-sec 90 --language es

# 3. Estimar hablantes y seleccionar clips
python3 scripts/select_clips.py --transcript-dir transcript_win --out-dir clips

# 4. Referencia compuesta 45–55 s por hablante (ajustar umbrales según clips/summary.json)
python3 scripts/build_composite.py --transcript-dir transcript_win --voice <id> \
  --pitch-min 185 --max-low 0.08 --target-sec 52 --out referencias/referencia_<id>

# 5. Registrar la voz y generar prueba
python3 scripts/register_voice.py --voice-id <id> --name "<Nombre>" --language es \
  --clip referencias/referencia_<id>.wav --transcript referencias/referencia_<id>.txt \
  --test-text "<frase de prueba>" --out prueba_<id>.wav
```

Idiomas soportados por el servicio: `en es fr de it pt ru zh ja ko`.

### Instalar el skill en tu agente

Copia la carpeta al directorio de skills de tu agente:

- **Kimi Work**: `~/Library/Application Support/kimi-desktop/daimon-share/daimon/skills/qwen3-voice-cloning/`
- **Claude Code**: `~/.claude/skills/qwen3-voice-cloning/`
- **Cursor / otros**: sigue la convención de skills de tu herramienta (carpeta con `SKILL.md`).

## Aprendizajes clave del pipeline

- **Una voz = una persona.** Mezclar hablantes en una referencia arruina el clon.
- **El pitch no separa voces del mismo registro** (p. ej. dos hombres). Usa curaduría por contenido y el *vocativo inverso*: quien **pronuncia** el nombre no es quien **lo tiene** ("Oye, Enrique…" lo dice Toño). Ancla con vocativos y expande por rango de pitch, revisando la lista final segmento por segmento.
- **Ojo con el sub-armónico**: voces rasposas pueden medir una octava abajo con autocorrelación; valida contra una referencia conocida de esa persona.
- **Referencias largas y variadas clonan mejor** (45–55 s compuestas, expresividad variada) que un clip corto y plano.
- **La transcripción de la referencia debe ser exacta** y terminar donde termina el audio; si sobra texto, el modelo "habla" las palabras restantes.
- Si el clon "no suena parecido", casi siempre es contaminación o referencia corta/plana: regenera la compuesta más estricta.

## Privacidad y uso responsable

Clona únicamente voces con consentimiento de su titular. El procesamiento es local, pero la responsabilidad de uso es tuya: no suplantes identidades ni publiques voces de terceros sin permiso.

## Licencia

MIT (ver `LICENSE`). Qwen3-TTS, OpenMontage y Whisper conservan sus propias licencias.
